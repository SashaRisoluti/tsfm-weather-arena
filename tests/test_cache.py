import threading
from collections.abc import Iterator
from datetime import date, datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
from omegaconf import DictConfig, OmegaConf

from arena.data import cache
from arena.data.sources import pardig, satweave

from .conftest import ABSENT, ALL_NAN, NO_MARKER, STAMP

# Row 35 of the synthetic grid is 43.5 N, the southern edge of the Alps box.
SOUTH_ROW = 35


def open_store(path: Path) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=False, chunks=None).load()


def expected_mm_per_hour(step_of_day: int, row_in_box: int) -> float:
    return (step_of_day + 1) * 0.1 + (SOUTH_ROW + row_in_box) * 0.001


def test_cache_root_is_per_region(synthetic: DictConfig, tmp_path: Path) -> None:
    assert cache.cache_root(synthetic) == tmp_path / "cache" / "alps"


def test_archive_is_cut_converted_and_written_once(
    synthetic: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = cache.cache_pardig_archive(synthetic)
    ds = open_store(path)
    assert ds.precipitation.shape == (96, 41, 81)
    assert ds.precipitation.attrs["units"] == "mm h-1"
    assert ds.latitude.values[0] == 43.5 and ds.longitude.values[-1] == 14.0
    np.testing.assert_allclose(
        ds.precipitation.values[0, 0, 0], expected_mm_per_hour(0, 0), rtol=1e-5
    )
    np.testing.assert_allclose(
        ds.precipitation.values[95, 40, 80], expected_mm_per_hour(95, 40), rtol=1e-5
    )

    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("archive was opened again")

    monkeypatch.setattr(pardig, "open_archive", fail)
    assert cache.cache_pardig_archive(synthetic) == path


def test_frame_day_is_cached_once(
    synthetic: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    day = date(2026, 1, 1)
    written = cache.cache_days(synthetic, "pardig", day, day)
    assert [p.name for p in written] == ["20260101.zarr"]
    ds = open_store(written[0])
    assert ds.precipitation.shape == (96, 41, 81)
    assert ds.attrs["missing_frames"] == 0
    assert ds.time.values[0] == np.datetime64("2026-01-01T00:00")
    assert ds.time.values[-1] == np.datetime64("2026-01-01T23:45")
    np.testing.assert_allclose(
        ds.precipitation.values[7, 3, 0], expected_mm_per_hour(7, 3), rtol=1e-5
    )

    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a frame was fetched again")

    monkeypatch.setattr(pardig, "open_frame", fail)
    assert cache.cache_days(synthetic, "pardig", day, day) == []


def test_day_with_missing_frames_is_not_cached_silently(synthetic: DictConfig) -> None:
    day = date(2026, 1, 2)
    assert cache.cache_days(synthetic, "pardig", day, day) == []
    assert not list(cache.cache_root(synthetic).rglob("20260102*"))

    written = cache.cache_days(synthetic, "pardig", day, day, allow_gaps=True)
    ds = open_store(written[0])
    assert ds.sizes["time"] == 96
    assert ds.attrs["missing_frames"] == 3
    for time in (ABSENT, ALL_NAN, NO_MARKER):
        assert bool(ds.precipitation.sel(time=time).isnull().all())
    assert int(ds.precipitation.isnull().all(("latitude", "longitude")).sum()) == 3


def test_step_with_one_band_missing_counts_as_a_gap(
    synthetic: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    day = date(2026, 1, 1)
    lost = datetime(2026, 1, 1, 7)
    real = satweave.open_frame

    def one_band_lost(cfg: DictConfig, time: datetime) -> xr.Dataset:
        frame = real(cfg, time)
        if time == lost:
            frame["mosaic_6200_nm"] = xr.full_like(frame.mosaic_6200_nm, np.nan)
        return frame

    monkeypatch.setattr(satweave, "open_frame", one_band_lost)
    assert cache.cache_days(synthetic, "satweave", day, day) == []

    written = cache.cache_days(synthetic, "satweave", day, day, allow_gaps=True)
    ds = open_store(written[0])
    assert ds.attrs["missing_frames"] == 1
    assert bool(ds.mosaic_6200_nm.sel(time=lost).isnull().all())
    # the band that did arrive is kept
    np.testing.assert_allclose(ds.mosaic_11200_nm.sel(time=lost).values, 257.0)


class Unreliable(SimpleHTTPRequestHandler):
    """Serves files, answering 503 for the paths listed in ``failing``."""

    failing: set[str] = set()

    def do_GET(self) -> None:
        if self.path in self.failing:
            self.send_error(503)
        else:
            super().do_GET()

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def served(remote: Path) -> Iterator[str]:
    """Address of an HTTP server over the synthetic stores."""
    handler = partial(Unreliable, directory=str(remote))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    Unreliable.failing.clear()


def test_failed_read_is_not_cached_as_a_gap(synthetic: DictConfig, served: str) -> None:
    frames = f"{served}/satweave/frames/{STAMP}/mosaic.zarr"
    OmegaConf.update(synthetic, "data.satweave.frame_url", frames)
    Unreliable.failing.add("/satweave/frames/20260101T070000/success")
    day = date(2026, 1, 1)
    assert cache.cache_days(synthetic, "satweave", day, day, allow_gaps=True) == []
    assert not list(cache.cache_root(synthetic).rglob("20260101*"))

    Unreliable.failing.clear()
    written = cache.cache_days(synthetic, "satweave", day, day, allow_gaps=True)
    assert open_store(written[0]).attrs["missing_frames"] == 0


def test_interrupted_write_leaves_no_store(
    synthetic: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    day = date(2026, 1, 1)
    real = xr.Dataset.to_zarr

    def crash(self: xr.Dataset, store: Any, *args: Any, **kwargs: Any) -> None:
        real(self, store, *args, **kwargs)
        raise RuntimeError("disk full")

    monkeypatch.setattr(xr.Dataset, "to_zarr", crash)
    with pytest.raises(RuntimeError):
        cache.cache_days(synthetic, "pardig", day, day)
    assert not list(cache.cache_root(synthetic).rglob("20260101*"))

    monkeypatch.setattr(xr.Dataset, "to_zarr", real)
    assert len(cache.cache_days(synthetic, "pardig", day, day)) == 1


def test_two_writers_of_one_store_do_not_share_a_temporary_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "store.zarr"
    first = xr.Dataset({"v": ("x", np.arange(4.0))})
    second = first + 10.0
    real = xr.Dataset.to_zarr
    targets: list[Path] = []

    def overtaken(self: xr.Dataset, store: Any, *args: Any, **kwargs: Any) -> None:
        targets.append(Path(store))
        real(self, store, *args, **kwargs)
        if len(targets) == 1:
            # another writer starts and finishes while this one is writing
            cache.write_atomic(second, path)

    monkeypatch.setattr(xr.Dataset, "to_zarr", overtaken)
    cache.write_atomic(first, path)
    assert targets[0] != targets[1]
    np.testing.assert_array_equal(open_store(path).v.values, second.v.values)
    assert [p.name for p in tmp_path.iterdir()] == ["store.zarr"]


def test_pardig_days_before_the_first_frame_are_refused(synthetic: DictConfig) -> None:
    with pytest.raises(ValueError):
        cache.cache_days(synthetic, "pardig", date(2025, 12, 31), date(2025, 12, 31))


def test_unknown_source_is_refused(synthetic: DictConfig) -> None:
    with pytest.raises(ValueError):
        cache.cache_days(synthetic, "radar", date(2026, 1, 1), date(2026, 1, 1))


def test_satweave_day_from_the_half_hourly_archive_is_hourly(
    synthetic: DictConfig,
) -> None:
    day = date(2025, 12, 31)
    ds = open_store(cache.cache_days(synthetic, "satweave", day, day)[0])
    assert sorted(ds.data_vars) == ["mosaic_11200_nm", "mosaic_6200_nm"]
    assert ds.mosaic_11200_nm.shape == (24, 41, 81)
    assert np.all(np.diff(ds.latitude.values) > 0)
    np.testing.assert_allclose(
        ds.mosaic_11200_nm.values[:, 0, 0], 250.0 + np.arange(24)
    )
    np.testing.assert_allclose(
        ds.mosaic_6200_nm.values[:, 40, 80], 220.0 + np.arange(24)
    )


def test_satweave_day_from_hourly_frames(synthetic: DictConfig) -> None:
    day = date(2026, 1, 1)
    ds = open_store(cache.cache_days(synthetic, "satweave", day, day)[0])
    assert ds.mosaic_6200_nm.shape == (24, 41, 81)
    assert ds.time.values[5] == np.datetime64(datetime(2026, 1, 1, 5))
    np.testing.assert_allclose(ds.mosaic_6200_nm.values[5], 225.0)


def test_open_cached_joins_archive_and_days_in_time_order(
    synthetic: DictConfig,
) -> None:
    cache.cache_days(synthetic, "pardig", date(2026, 1, 1), date(2026, 1, 1))
    cache.cache_pardig_archive(synthetic)
    with cache.open_cached(synthetic, "pardig") as ds:
        assert ds.sizes == {"time": 192, "latitude": 41, "longitude": 81}
        assert bool(np.all(np.diff(ds.time.values) == np.timedelta64(15, "m")))
        assert ds.precipitation.attrs["units"] == "mm h-1"


def test_open_cached_without_data_raises(synthetic: DictConfig) -> None:
    with pytest.raises(FileNotFoundError):
        cache.open_cached(synthetic, "satweave")
