from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
import xarray as xr
from omegaconf import DictConfig, OmegaConf

from arena.data import cache
from arena.data.region import Region
from arena.data.sources import elevation

ALPS = Region("alps", lon_min=6.0, lon_max=14.0, lat_min=43.5, lat_max=47.5)


class FakeApi:
    """Answers like the elevation service, with a known surface."""

    def __init__(self) -> None:
        self.sizes: list[int] = []

    def __call__(self, url: str) -> dict[str, Any]:
        query = parse_qs(urlparse(url).query)
        lat = np.array(query["latitude"][0].split(","), dtype="float64")
        lon = np.array(query["longitude"][0].split(","), dtype="float64")
        assert lat.size == lon.size
        self.sizes.append(lat.size)
        return {"elevation": list(lat**2 + lon)}


def test_fetch_samples_the_centre_of_each_cell_by_default(
    synthetic: DictConfig,
) -> None:
    api = FakeApi()
    out = elevation.fetch(synthetic, ALPS, get_json=api)
    assert out.dims == ("latitude", "longitude")
    assert out.shape == (41, 81)
    assert out.attrs["units"] == "m"
    assert (out.latitude.values[0], out.longitude.values[-1]) == (43.5, 14.0)
    expected = out.latitude**2 + out.longitude
    np.testing.assert_allclose(out.values, expected.values, rtol=1e-9)
    assert max(api.sizes) <= 100
    assert sum(api.sizes) == 41 * 81


def test_fetch_can_average_sub_samples_of_each_cell(synthetic: DictConfig) -> None:
    OmegaConf.update(synthetic, "data.elevation.subsamples", 3)
    api = FakeApi()
    out = elevation.fetch(synthetic, ALPS, get_json=api)
    # Sub-samples sit a third of a cell apart, so the mean of lat**2 over a
    # cell exceeds the centre value by the variance of those offsets.
    spread = (2 / 3) * (0.1 / 3) ** 2
    expected = out.latitude**2 + spread + out.longitude
    np.testing.assert_allclose(out.values, expected.values, rtol=1e-9)
    assert sum(api.sizes) == 41 * 81 * 9


def test_fetch_waits_and_retries_when_rate_limited(
    synthetic: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeApi()
    refusals = [elevation.RateLimited(5.0), elevation.RateLimited(7.0)]
    waits: list[float] = []

    def limited(url: str) -> dict[str, Any]:
        if refusals:
            raise refusals.pop(0)
        return api(url)

    monkeypatch.setattr(elevation.time, "sleep", waits.append)
    out = elevation.fetch(synthetic, ALPS, get_json=limited)
    assert waits == [6.0, 8.0]
    assert out.shape == (41, 81) and sum(api.sizes) == 41 * 81


def test_fetch_gives_up_when_the_wait_is_too_long(
    synthetic: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def exhausted(url: str) -> dict[str, Any]:
        raise elevation.RateLimited(3600.0)

    monkeypatch.setattr(elevation.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="request limit"):
        elevation.fetch(synthetic, ALPS, get_json=exhausted)


def test_fetch_rejects_a_short_answer(synthetic: DictConfig) -> None:
    with pytest.raises(ValueError):
        elevation.fetch(synthetic, ALPS, get_json=lambda url: {"elevation": [1.0]})


def test_elevation_is_cached_once(
    synthetic: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake(cfg: DictConfig, region: Region) -> xr.DataArray:
        calls.append(region)
        return xr.DataArray(
            np.ones((2, 2)),
            dims=("latitude", "longitude"),
            coords={"latitude": [43.5, 43.6], "longitude": [6.0, 6.1]},
            attrs={"units": "m"},
        )

    monkeypatch.setattr(elevation, "fetch", fake)
    path = cache.cache_elevation(synthetic)
    assert path == cache.cache_root(synthetic) / "static" / "elevation.zarr"
    assert cache.cache_elevation(synthetic) == path
    assert len(calls) == 1
    with xr.open_zarr(path, consolidated=False) as ds:
        assert ds.elevation.shape == (2, 2)
