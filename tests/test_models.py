from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from arena.config import load_config
from arena.eval.quantiles import LEVELS
from arena.models.base import Forecaster
from arena.models.baselines.climatology import Climatology
from arena.models.baselines.extrapolation import Extrapolation, motion
from arena.models.baselines.persistence import Persistence
from arena.models.reference.aifs_ens import AifsReference

ISSUE = datetime(2025, 7, 15, 12, 0)
MINUTES = [timedelta(minutes=m) for m in (15, 30, 60)]


def grid_frames(times: pd.DatetimeIndex, values: np.ndarray) -> xr.DataArray:
    lat = np.round(45.0 + 0.1 * np.arange(values.shape[1]), 6)
    lon = np.round(7.0 + 0.1 * np.arange(values.shape[2]), 6)
    return xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": lat, "longitude": lon},
    )


def blob(rows: int, cols: int, row: int, col: int) -> np.ndarray:
    field = np.zeros((rows, cols))
    field[row - 1 : row + 2, col - 1 : col + 2] = 4.0
    field[row, col] = 8.0
    return field


def test_baselines_are_forecasters() -> None:
    for model in (Persistence(), Climatology(), Extrapolation()):
        assert isinstance(model, Forecaster)


def test_persistence_repeats_the_last_visible_value() -> None:
    times = pd.date_range("2025-07-15T11:00", "2025-07-15T12:00", freq="15min")
    context = xr.DataArray(
        np.arange(10.0).reshape(5, 2),
        dims=("time", "pixel"),
        coords={"time": times},
    )
    out = Persistence().predict(context, None, ISSUE, MINUTES)
    assert out.dims == ("quantile", "lead", "pixel")
    np.testing.assert_allclose(out["quantile"].values, LEVELS)
    np.testing.assert_allclose(out.values, np.broadcast_to([8.0, 9.0], out.shape))
    assert list(out.lead.values) == [np.timedelta64(m) for m in MINUTES]


def test_a_context_reaching_past_the_issue_time_is_refused() -> None:
    times = pd.date_range("2025-07-15T11:00", "2025-07-15T12:15", freq="15min")
    context = xr.DataArray(
        np.zeros((6, 2)), dims=("time", "pixel"), coords={"time": times}
    )
    with pytest.raises(ValueError, match="after the issue time"):
        Persistence().predict(context, None, ISSUE, MINUTES)


def test_climatology_gives_the_quantiles_of_its_month_and_hour() -> None:
    times = pd.date_range("2016-07-01", "2016-07-31T23:00", freq="1h")
    # each hour of the day holds the values day 1 .. 31 plus 100 x hour
    values = (times.day + 100 * times.hour).to_numpy(dtype="float64")[:, None]
    history = xr.DataArray(values, dims=("time", "pixel"), coords={"time": times})
    model = Climatology()
    model.fit(history)
    context = history.isel(time=slice(-4, None))
    out = model.predict(
        context, None, datetime(2025, 7, 15, 11, 0), [timedelta(hours=2)]
    )
    expected = np.quantile(np.arange(1.0, 32.0), LEVELS) + 1300.0
    np.testing.assert_allclose(out.isel(lead=0, pixel=0).values, expected)
    with pytest.raises(KeyError):
        model.predict(context, None, datetime(2025, 8, 15, 11, 0), [timedelta(hours=2)])


def test_motion_finds_the_shift_between_two_frames() -> None:
    before = blob(20, 30, 10, 10)
    after = blob(20, 30, 11, 13)
    assert motion(before, after) == (1, 3)
    assert motion(np.zeros((20, 30)), np.zeros((20, 30))) == (0, 0)


def test_extrapolation_moves_rain_along_the_recent_motion() -> None:
    # one cell east every 15 minutes over the last hour
    times = pd.date_range("2025-07-15T11:00", "2025-07-15T12:00", freq="15min")
    frames = np.stack([blob(20, 30, 10, 8 + k) for k in range(5)])
    context = grid_frames(times, frames)
    out = Extrapolation().predict(context, None, ISSUE, [timedelta(minutes=30)])
    median = out.sel(quantile=0.5).isel(lead=0).values
    np.testing.assert_allclose(median, blob(20, 30, 10, 14))


def test_extrapolation_without_rain_or_an_hour_of_history_is_persistence() -> None:
    times = pd.date_range("2025-07-15T11:45", "2025-07-15T12:00", freq="15min")
    context = grid_frames(times, np.stack([blob(20, 30, 10, 8), blob(20, 30, 10, 9)]))
    out = Extrapolation().predict(context, None, ISSUE, [timedelta(minutes=30)])
    np.testing.assert_allclose(
        out.sel(quantile=0.5).isel(lead=0).values, blob(20, 30, 10, 9)
    )


def forecast_runs() -> xr.Dataset:
    lat = np.round(43.5 + 0.25 * np.arange(17), 6)
    lon = np.round(6.0 + 0.25 * np.arange(33), 6)
    inits = pd.date_range("2026-06-01T00:00", periods=2, freq="12h")
    steps = np.arange(6, 61, 6)
    shape = (2, 3, steps.size, lat.size, lon.size)
    tp = np.broadcast_to(np.arange(1.0, 4.0)[None, :, None, None, None], shape)
    return xr.Dataset(
        {"tp": (("init_time", "member", "step", "latitude", "longitude"), tp.copy())},
        coords={
            "init_time": inits,
            "member": [0, 1, 2],
            "step": steps,
            "latitude": lat,
            "longitude": lon,
        },
    )


def test_aifs_reference_serves_stored_runs_as_quantiles_at_the_context_points() -> None:
    model = AifsReference(forecast_runs(), load_config())
    assert isinstance(model, Forecaster)
    context = xr.DataArray(
        np.zeros((1, 2)),
        dims=("time", "pixel"),
        coords={
            "time": [np.datetime64("2026-06-01T23:00")],
            "latitude": ("pixel", [45.0, 46.3]),
            "longitude": ("pixel", [7.0, 12.1]),
        },
    )
    leads = [timedelta(hours=6), timedelta(hours=48)]
    out = model.predict(context, None, datetime(2026, 6, 2, 0), leads)
    assert out.dims == ("quantile", "lead", "pixel")
    np.testing.assert_allclose(out.sel(quantile=0.5).values, 2.0)
    np.testing.assert_allclose(out.sel(quantile=0.05).values, 1.1)
    with pytest.raises(ValueError, match="6 h"):
        model.predict(context, None, datetime(2026, 6, 2, 0), [timedelta(hours=3)])


def test_extrapolation_ignores_an_implausibly_fast_motion() -> None:
    # 20 cells east in an hour is about 160 km/h at 45 N: treated as noise
    times = pd.date_range("2025-07-15T11:00", "2025-07-15T12:00", freq="15min")
    frames = np.stack([blob(20, 40, 10, 5)] * 4 + [blob(20, 40, 10, 25)])
    context = grid_frames(times, frames)
    out = Extrapolation().predict(context, None, ISSUE, [timedelta(minutes=30)])
    np.testing.assert_allclose(out.sel(quantile=0.5).isel(lead=0).values, frames[-1])
