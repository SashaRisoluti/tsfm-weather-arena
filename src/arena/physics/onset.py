"""Precipitation onsets and lagged correlation with a covariate.

Used to check that cloud-top cooling precedes rain, the physical case for
the satellite covariates.
"""

from collections.abc import Iterable

import xarray as xr


def onsets(
    rate: xr.DataArray, threshold: float, dry: float, dry_hours: int
) -> xr.DataArray:
    """True where an hourly ``rate`` (mm/h) reaches ``threshold`` after
    ``dry_hours`` hours all below ``dry``. A missing hour is not dry."""
    # On a regular hourly axis, so that an hour absent from the data is missing.
    hourly = rate.resample(time="1h").asfreq()
    before = hourly.shift(time=1).rolling(time=dry_hours).max()
    started = (hourly >= threshold) & (before < dry)
    return started.sel(time=rate.time).astype(bool)


def lagged_correlation(
    x: xr.DataArray, y: xr.DataArray, lags: Iterable[int]
) -> xr.DataArray:
    """Pearson correlation of x(t - lag) with y(t), pooled over all dimensions.

    ``lag`` counts hours; a positive lag means x leads y. Both series are put
    on a regular hourly axis first, and pairs with a missing value are skipped.
    """
    lags = list(lags)
    x = x.resample(time="1h").asfreq()
    y = y.astype(float).resample(time="1h").asfreq()
    values = [float(xr.corr(x.shift(time=lag), y)) for lag in lags]
    return xr.DataArray(values, dims="lag", coords={"lag": lags})
