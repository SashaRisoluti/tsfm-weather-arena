"""Scores of quantile forecasts against observations.

Every probabilistic score is computed from the quantiles at the common levels
(``eval.quantiles.LEVELS``), so all models are scored the same way. The
forecast distribution between quantiles is linear; outside the outermost
levels its CDF is flat (0.05 below the lowest quantile, 0.95 above the
highest).
"""

import numpy as np
import pandas as pd
import xarray as xr


def crps_quantile(q: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    """CRPS approximated as twice the mean pinball loss over the levels."""
    tau = q["quantile"]
    pinball = (q - obs) * ((obs < q).astype("float64") - tau)
    return 2 * pinball.mean("quantile").drop_vars("quantile", errors="ignore")


def crps_ensemble(members: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    """Exact CRPS of an ensemble: E|X - y| - E|X - X'| / 2."""
    x = members.transpose(..., "member")
    m = x.sizes["member"]
    error = abs(x - obs).mean("member")
    weights = 2 * np.arange(1, m + 1) - m - 1
    # E|X - X'| from the order statistics, without the m x m differences.
    spread = (np.sort(x.values, axis=-1) * weights).sum(axis=-1) * 2 / m**2
    return error - 0.5 * x.isel(member=0, drop=True).copy(data=spread)


def mae_median(q: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    return abs(q.sel(quantile=0.5, drop=True) - obs)


def _cdf(q: xr.DataArray, x: xr.DataArray | float) -> xr.DataArray:
    """Forecast CDF at ``x``, linear between quantiles, flat outside them."""
    k = q.sizes["quantile"]
    tau = q["quantile"].values
    below = (q <= x).sum("quantile")
    lo = (below - 1).clip(0, k - 2)
    hi = lo + 1
    q_lo, q_hi = q.isel(quantile=lo), q.isel(quantile=hi)
    t_lo = q["quantile"].isel(quantile=lo)
    t_hi = q["quantile"].isel(quantile=hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = t_lo + (x - q_lo) / (q_hi - q_lo) * (t_hi - t_lo)
    cdf = xr.where(below == 0, tau[0], xr.where(below == k, tau[-1], inner))
    return cdf.drop_vars("quantile", errors="ignore")


def exceedance_probability(q: xr.DataArray, threshold: float) -> xr.DataArray:
    """Forecast probability that the amount is above ``threshold``."""
    return 1.0 - _cdf(q, threshold)


def brier(p: xr.DataArray, observed: xr.DataArray) -> xr.DataArray:
    return (p - observed.astype("float64")) ** 2


def categorical(
    forecast: xr.DataArray, observed: xr.DataArray, dim: str | list[str]
) -> xr.Dataset:
    """CSI, POD and FAR of yes/no forecasts, from counts summed over ``dim``."""
    hits = (forecast & observed).sum(dim)
    misses = (~forecast & observed).sum(dim)
    false_alarms = (forecast & ~observed).sum(dim)
    with np.errstate(divide="ignore", invalid="ignore"):
        return xr.Dataset(
            {
                "csi": hits / (hits + misses + false_alarms),
                "pod": hits / (hits + misses),
                "far": false_alarms / (hits + false_alarms),
                "hits": hits,
                "misses": misses,
                "false_alarms": false_alarms,
            }
        )


def coverage(
    q: xr.DataArray, obs: xr.DataArray, intervals: tuple[float, ...] = (0.5, 0.9)
) -> xr.DataArray:
    """Whether ``obs`` lies in each central interval of the forecast."""
    inside = []
    for width in intervals:
        lower = q.sel(quantile=round((1 - width) / 2, 2), drop=True)
        upper = q.sel(quantile=round((1 + width) / 2, 2), drop=True)
        inside.append((obs >= lower) & (obs <= upper))
    return xr.concat(inside, dim="interval").assign_coords(interval=list(intervals))


def pit(q: xr.DataArray, obs: xr.DataArray, rng: np.random.Generator) -> xr.DataArray:
    """Probability integral transform F(obs).

    A dry observation falls on the forecast's mass at zero, so its PIT is
    drawn uniformly from [0, F(0)] (randomised PIT).
    """
    at_obs = _cdf(q, obs)
    at_zero = _cdf(q, 0.0)
    at_zero, at_obs = xr.broadcast(at_zero, at_obs)
    draw = at_zero.copy(data=rng.uniform(size=at_zero.shape)) * at_zero
    return xr.where(obs > 0, at_obs, draw)


def reliability(
    p: xr.DataArray, observed: xr.DataArray, bins: int = 10
) -> pd.DataFrame:
    """Mean forecast probability, observed frequency and count per bin."""
    prob = p.values.ravel()
    seen = observed.values.ravel().astype("float64")
    index = np.minimum((prob * bins).astype(int), bins - 1)
    frame = pd.DataFrame({"bin": index, "forecast": prob, "observed": seen})
    table = frame.groupby("bin").agg(
        forecast=("forecast", "mean"),
        observed=("observed", "mean"),
        count=("observed", "size"),
    )
    return table.reindex(range(bins))


def _box_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Mean over a window x window box around each cell, zeros outside."""
    half = window // 2
    padded = np.pad(values, half + 1)
    total = padded.cumsum(axis=0).cumsum(axis=1)
    rows, cols = values.shape
    box = (
        total[window : window + rows, window : window + cols]
        - total[:rows, window : window + cols]
        - total[window : window + rows, :cols]
        + total[:rows, :cols]
    )
    return box / window**2


def fss(
    forecast: xr.DataArray, observed: xr.DataArray, threshold: float, window: int
) -> float:
    """Fractions skill score of two 2-D fields for one threshold and an odd
    window size in cells."""
    if window % 2 == 0:
        raise ValueError("window must be an odd number of cells")
    fc = _box_mean((forecast.values >= threshold).astype("float64"), window)
    ob = _box_mean((observed.values >= threshold).astype("float64"), window)
    reference = np.mean(fc**2) + np.mean(ob**2)
    if reference == 0:
        return float("nan")
    return float(1 - np.mean((fc - ob) ** 2) / reference)
