from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from arena.config import load_config
from arena.data.align import nwp_as_of, past_as_of, six_hour_totals

CFG = load_config()


def rates(start: str, periods: int) -> xr.DataArray:
    """A 15-minute series whose rate equals the frame number, in mm/h."""
    time = pd.date_range(start, periods=periods, freq="15min")
    return xr.DataArray(
        np.arange(periods, dtype="float64"), dims="time", coords={"time": time}
    )


def test_six_hour_totals_sum_the_frames_after_the_window_start() -> None:
    # Frames 00:00 .. 12:00; the window ending 06:00 holds 00:15 .. 06:00.
    out = six_hour_totals(rates("2026-06-01T00:00", 49), step_minutes=15)
    assert out.attrs["units"] == "mm"
    np.testing.assert_allclose(
        out.sel(time="2026-06-01T06:00").item(), sum(range(1, 25)) * 0.25
    )
    np.testing.assert_allclose(
        out.sel(time="2026-06-01T12:00").item(), sum(range(25, 49)) * 0.25
    )


def test_six_hour_total_with_a_missing_frame_is_missing() -> None:
    series = rates("2026-06-01T00:00", 49)
    gappy = series.drop_sel(time=np.datetime64("2026-06-01T03:00"))
    out = six_hour_totals(gappy, step_minutes=15)
    assert np.isnan(out.sel(time="2026-06-01T06:00").item())
    assert not np.isnan(out.sel(time="2026-06-01T12:00").item())
    # the 00:00 window has only one frame and is missing too
    assert np.isnan(out.sel(time="2026-06-01T00:00").item())


def test_past_as_of_never_returns_what_is_not_yet_published() -> None:
    series = rates("2026-06-01T00:00", 200)
    for minutes in range(0, 48 * 60, 5):
        issue = datetime(2026, 6, 1) + timedelta(minutes=minutes)
        for lag in (0, 60, 90):
            seen = past_as_of(series, issue, lag)
            latest = np.datetime64(issue - timedelta(minutes=lag))
            assert seen.time.size == 0 or seen.time.values[-1] <= latest
            assert seen.time.size == int((series.time.values <= latest).sum())


def forecast_runs() -> xr.Dataset:
    inits = pd.date_range("2026-06-01T00:00", periods=8, freq="12h")
    steps = np.arange(6, 61, 6)
    values = (
        inits.values.astype("datetime64[h]").astype("int64")[:, None] * 1000
        + steps[None, :]
    ).astype("float64")
    return xr.Dataset(
        {"tp": (("init_time", "member", "step"), values[:, None, :].repeat(2, 1))},
        coords={"init_time": inits, "member": [0, 1], "step": steps},
    )


def test_nwp_as_of_uses_the_newest_run_already_disseminated() -> None:
    runs = forecast_runs()
    leads = list(CFG.data.align.regime_b.leads_hours)
    for hours in range(24, 84):
        issue = datetime(2026, 6, 1) + timedelta(hours=hours)
        newest = runs.init_time.values[
            runs.init_time.values <= np.datetime64(issue - timedelta(hours=12))
        ][-1]
        try:
            out = nwp_as_of(runs, issue, leads, CFG)
        except KeyError:
            # leads beyond the archived steps (issues not at 00 or 12 UTC)
            assert (
                issue - newest.astype("datetime64[s]").item()
            ).total_seconds() / 3600 + 48 > 60
            continue
        assert out.init_time.values == newest
        assert list(out.lead.values) == leads
        issued = out.init_time.values + np.timedelta64(12, "h")
        assert issued <= np.datetime64(issue)
        valid_end = out.init_time.values + out.step.values.astype("timedelta64[h]")
        np.testing.assert_array_equal(
            valid_end, np.datetime64(issue) + np.array(leads).astype("timedelta64[h]")
        )


def test_nwp_as_of_at_00_utc_uses_the_run_twelve_hours_earlier() -> None:
    out = nwp_as_of(forecast_runs(), datetime(2026, 6, 2, 0), [6, 48], CFG)
    assert out.init_time.values == np.datetime64("2026-06-01T12:00")
    assert list(out.step.values) == [18, 60]
    assert out.tp.dims == ("member", "lead")


def test_nwp_as_of_refuses_an_issue_between_whole_hours() -> None:
    with pytest.raises(ValueError, match="whole hours"):
        nwp_as_of(forecast_runs(), datetime(2026, 6, 2, 0, 30), [6], CFG)


def test_nwp_as_of_refuses_a_run_that_is_not_cached() -> None:
    with pytest.raises(KeyError, match="2026-06-05"):
        nwp_as_of(forecast_runs(), datetime(2026, 6, 6, 0), [6], CFG)
