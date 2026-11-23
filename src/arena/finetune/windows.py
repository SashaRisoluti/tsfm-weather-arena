"""Series, covariates and transforms for time-series models.

A model reads one series per evaluation pixel: the pixel alone, or the pixel
and its eight neighbours (a crude proxy for advection). Covariates published
on another cadence are brought onto the model's time steps by an as-of join.
"""

import numpy as np
import pandas as pd
import xarray as xr

# Centre first, then the eight neighbours row by row.
_OFFSETS = [(0, 0)] + [
    (r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0)
]


def _positions(axis: np.ndarray, values: np.ndarray, name: str) -> np.ndarray:
    index = np.searchsorted(axis, values)
    index = np.clip(index, 0, axis.size - 1)
    if not np.allclose(axis[index], values):
        raise ValueError(f"pixels are not on the grid's {name} nodes")
    return index


def pixel_series(
    field: xr.DataArray, pixels: pd.DataFrame, neighbours: bool
) -> np.ndarray:
    """Series at each pixel, shape (pixel, variate, ...), where ``...`` are
    the dimensions of ``field`` before latitude and longitude.

    With ``neighbours`` the variates are the pixel and its eight neighbours;
    neighbours outside the grid repeat the nearest edge cell.
    """
    grid = field.transpose(..., "latitude", "longitude").values
    rows = _positions(field.latitude.values, pixels.latitude.to_numpy(), "latitude")
    cols = _positions(field.longitude.values, pixels.longitude.to_numpy(), "longitude")
    offsets = _OFFSETS if neighbours else _OFFSETS[:1]
    n_rows, n_cols = grid.shape[-2:]
    variates = [
        grid[..., np.clip(rows + dr, 0, n_rows - 1), np.clip(cols + dc, 0, n_cols - 1)]
        for dr, dc in offsets
    ]
    # (variate, ..., pixel) -> (pixel, variate, ...)
    return np.moveaxis(np.stack(variates, axis=0), -1, 0)


def to_steps(
    features: xr.DataArray,
    steps: pd.DatetimeIndex,
    lag_minutes: int,
    max_age_minutes: int = 60,
) -> xr.DataArray:
    """Features on ``steps``: at step s, the newest value published by s.

    A value at time u is published at u + ``lag_minutes``. Steps whose newest
    published value is older than ``max_age_minutes`` are missing.
    """
    lag = pd.Timedelta(minutes=lag_minutes)
    seen = features.reindex(
        time=steps - lag,
        method="ffill",
        tolerance=pd.Timedelta(minutes=max_age_minutes),
    )
    return seen.assign_coords(time=steps)


def transform(values: np.ndarray, name: str) -> np.ndarray:
    if name == "log1p":
        return np.log1p(np.clip(values, 0.0, None))
    if name == "none":
        return values
    raise ValueError(f"unknown transform {name!r}")


def inverse(values: np.ndarray, name: str) -> np.ndarray:
    if name == "log1p":
        return np.expm1(values)
    if name == "none":
        return values
    raise ValueError(f"unknown transform {name!r}")
