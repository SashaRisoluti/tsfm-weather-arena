"""Common quantile levels shared by every model.

Models publish different outputs (9 or 21 quantiles, or 51 ensemble members).
Before scoring, each is mapped to the same 19 levels, 0.05 to 0.95, by the
same rule:

- given quantiles: linear interpolation in probability between the given
  levels, and linear extrapolation from the two nearest levels beyond them;
  values are then clipped at 0 and sorted, so the result is a valid set of
  quantiles of a non-negative amount;
- ensemble members: empirical quantiles with linear interpolation between
  order statistics (numpy's default method).
"""

import numpy as np
import xarray as xr

LEVELS = np.round(np.arange(1, 20) * 0.05, 2)


def harmonise(q: xr.DataArray) -> xr.DataArray:
    """Map quantiles on any levels (dimension ``quantile``) to ``LEVELS``."""
    levels = q["quantile"].values.astype("float64")
    order = np.argsort(levels)
    levels = levels[order]
    values = q.isel(quantile=order).transpose(..., "quantile").values
    seg = np.clip(np.searchsorted(levels, LEVELS, side="right") - 1, 0, levels.size - 2)
    lo, hi = levels[seg], levels[seg + 1]
    slope = (values[..., seg + 1] - values[..., seg]) / (hi - lo)
    out = np.sort(np.clip(values[..., seg] + (LEVELS - lo) * slope, 0.0, None), axis=-1)
    others = [dim for dim in q.dims if dim != "quantile"]
    coords = {name: c for name, c in q.coords.items() if "quantile" not in c.dims}
    result = xr.DataArray(
        out,
        dims=(*others, "quantile"),
        coords={**coords, "quantile": LEVELS},
        attrs=q.attrs,
    )
    return result.transpose("quantile", *others)


def from_ensemble(members: xr.DataArray) -> xr.DataArray:
    """Quantiles at ``LEVELS`` of an ensemble with dimension ``member``."""
    return members.quantile(LEVELS, dim="member", keep_attrs=True)
