from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from arena.physics import base
from arena.physics.radiative import IR, WV
from arena.physics.solar import cos_solar_zenith

LAT = np.array([45.0, 45.1])
LON = np.array([7.0, 7.1, 7.2])


def satellite(ir: list[float], wv: list[float], times: list[str]) -> xr.Dataset:
    shape = (len(times), LAT.size, LON.size)

    def cube(values: list[float]) -> np.ndarray:
        return np.broadcast_to(np.array(values)[:, None, None], shape).copy()

    return xr.Dataset(
        {
            IR: (("time", "latitude", "longitude"), cube(ir), {"units": "K"}),
            WV: (("time", "latitude", "longitude"), cube(wv), {"units": "K"}),
        },
        coords={"time": pd.to_datetime(times), "latitude": LAT, "longitude": LON},
    )


HOURS = ["2025-07-15T10:00", "2025-07-15T11:00", "2025-07-15T12:00", "2025-07-15T14:00"]


def test_radiative_covariates_by_hand() -> None:
    ds = satellite([280.0, 270.0, 250.0, 240.0], [230.0, 228.0, 225.0, 220.0], HOURS)
    out = base.covariates(ds, ["radiative"])
    point = out.isel(latitude=0, longitude=0)
    np.testing.assert_allclose(point.sel(feature="bt_diff").values, [50, 42, 25, 20])
    # backward differences per hour; 14:00 follows a gap, so it has none
    rate = point.sel(feature="dbt_dt").values
    assert np.isnan(rate[0]) and np.isnan(rate[3])
    np.testing.assert_allclose(rate[1:3], [-10.0, -20.0])
    assert not point.known_future.sel(feature="dbt_dt")


def test_cooling_rate_at_a_time_uses_no_later_frame() -> None:
    ds = satellite(
        [280.0, 270.0, 250.0, 240.0], [230.0] * 4, HOURS[:3] + ["2025-07-15T13:00"]
    )
    changed = ds.copy(deep=True)
    changed[IR][-1] = 100.0
    before = base.covariates(ds, ["radiative"]).sel(feature="dbt_dt")
    after = base.covariates(changed, ["radiative"]).sel(feature="dbt_dt")
    xr.testing.assert_equal(before.isel(time=slice(0, 3)), after.isel(time=slice(0, 3)))


class CloudTopDummy:
    """Stands in for a provider added later by another module."""

    name: ClassVar[str] = "cloud_top_dummy"
    requires: ClassVar[tuple[str, ...]] = (IR,)
    known_future: ClassVar[bool] = False

    def compute(self, ds: xr.Dataset) -> xr.Dataset:
        return xr.Dataset({"cloud_top_km": (ds[IR] - 200.0) / 10.0})


def test_a_provider_registered_later_flows_into_the_covariates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base.provider_names()  # load the built-in providers before isolating the registry
    monkeypatch.setattr(base, "PROVIDERS", dict(base.PROVIDERS))
    base.register(CloudTopDummy)
    ds = satellite([280.0, 270.0], [230.0, 228.0], HOURS[:2])
    out = base.covariates(ds, ["radiative", "solar", "cloud_top_dummy"])
    assert out.dims == ("feature", "time", "latitude", "longitude")
    assert "cloud_top_km" in out.feature.values
    np.testing.assert_allclose(out.sel(feature="cloud_top_km").values[1], 7.0)
    flags = dict(zip(out.feature.values, out.known_future.values, strict=True))
    assert not flags["cloud_top_km"]
    assert flags["cos_zenith"]


def test_a_name_already_taken_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    base.provider_names()
    monkeypatch.setattr(base, "PROVIDERS", dict(base.PROVIDERS))

    class Impostor(CloudTopDummy):
        name: ClassVar[str] = "solar"

    with pytest.raises(ValueError, match="solar"):
        base.register(Impostor)


def test_unknown_provider_and_missing_input_are_refused() -> None:
    ds = satellite([280.0], [230.0], HOURS[:1])
    with pytest.raises(KeyError, match="nope"):
        base.covariates(ds, ["nope"])
    with pytest.raises(ValueError, match=WV):
        base.covariates(ds.drop_vars(WV), ["radiative"])


def test_providers_are_found_by_name() -> None:
    assert {"radiative", "solar"} <= set(base.provider_names())


def minimum_zenith(day: str, lat: float) -> tuple[float, pd.Timestamp]:
    times = pd.date_range(f"{day}T00:00", periods=24 * 60, freq="1min")
    cosz = cos_solar_zenith(times.values, np.array([lat]), np.array([0.0]))[:, 0, 0]
    best = int(np.argmax(cosz))
    return float(np.degrees(np.arccos(cosz[best]))), times[best]


@pytest.mark.parametrize(
    ("day", "lat", "expected"),
    [
        ("2025-06-21", 45.0, 45.0 - 23.44),  # summer solstice
        ("2025-12-21", 45.0, 45.0 + 23.44),  # winter solstice
        ("2025-03-20", 45.0, 45.0),  # equinox
        ("2025-03-20", 0.0, 0.0),
    ],
)
def test_noon_zenith_matches_the_textbook(
    day: str, lat: float, expected: float
) -> None:
    zenith, when = minimum_zenith(day, lat)
    assert zenith == pytest.approx(expected, abs=0.5)
    # at the Greenwich meridian solar noon is within a quarter hour of 12 UTC
    assert abs((when - pd.Timestamp(f"{day}T12:00")).total_seconds()) < 15 * 60


def test_solar_noon_moves_with_longitude_and_the_equation_of_time() -> None:
    times = pd.date_range("2025-11-03T00:00", periods=24 * 60, freq="1min")
    east = cos_solar_zenith(times.values, np.array([45.0]), np.array([90.0]))[:, 0, 0]
    greenwich = cos_solar_zenith(times.values, np.array([45.0]), np.array([0.0]))[
        :, 0, 0
    ]
    # the equation of time is about +16.4 min in early November
    noon = times[int(np.argmax(greenwich))]
    assert pd.Timestamp("2025-11-03T11:41") <= noon <= pd.Timestamp("2025-11-03T11:47")
    # 90 degrees east is six hours earlier
    assert times[int(np.argmax(east))] == noon - pd.Timedelta(hours=6)


def test_solar_covariates_are_known_in_advance_and_bounded() -> None:
    ds = satellite(
        [280.0] * 3,
        [230.0] * 3,
        ["2025-07-15T00:00", "2025-07-15T06:00", "2025-07-15T12:00"],
    )
    out = base.covariates(ds, ["solar"])
    assert bool(out.known_future.all())
    cosz = out.sel(feature="cos_zenith")
    assert float(cosz.isel(time=0).max()) < 0 < float(cosz.isel(time=2).min())
    for name in ("doy_sin", "doy_cos", "hour_sin", "hour_cos"):
        assert float(abs(out.sel(feature=name)).max()) <= 1.0
    np.testing.assert_allclose(out.sel(feature="hour_sin").isel(time=1).values, 1.0)
