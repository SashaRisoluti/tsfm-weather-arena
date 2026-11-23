import numpy as np
import pandas as pd
import xarray as xr

from arena.physics.onset import lagged_correlation, onsets


def hourly(values: list[float]) -> xr.DataArray:
    time = pd.date_range("2025-07-15", periods=len(values), freq="1h")
    return xr.DataArray(np.array(values), dims="time", coords={"time": time})


def test_onset_needs_rain_after_dry_hours() -> None:
    rate = hourly(
        [0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.05, 1.5, 0.0, 0.5, 1.2, np.nan, 2.0]
    )
    got = onsets(rate, threshold=1.0, dry=0.1, dry_hours=2)
    # 03:00 starts rain; 04:00 continues it; 08:00 follows two dry hours;
    # 11:00 follows a light hour; 13:00 follows a missing hour
    assert np.flatnonzero(got.values).tolist() == [3, 8]
    assert got.dtype == bool


def test_lagged_correlation_peaks_at_the_lead_of_x() -> None:
    rng = np.random.default_rng(1)
    x = hourly(list(rng.normal(size=500)))
    y = x.shift(time=3)  # y follows x by three hours
    corr = lagged_correlation(x, y, lags=(lag for lag in range(-2, 6)))
    assert list(corr.lag.values) == list(range(-2, 6))
    assert int(corr.idxmax("lag")) == 3
    np.testing.assert_allclose(corr.sel(lag=3), 1.0)
    assert abs(float(corr.sel(lag=0))) < 0.2
