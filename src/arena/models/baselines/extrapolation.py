"""Lagrangian persistence: the last frame moved along the recent motion.

One motion vector per issue time, found by phase correlation between the
frames one hour and zero hours before the issue. The last frame is shifted
by that motion times the lead, in whole cells; cells with no upstream data
keep their last value. Rain neither grows nor decays.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import ClassVar

import numpy as np
import xarray as xr

from arena.models.base import check_context, lead_coord, point_quantiles

# Below this rate (mm/h) everywhere, a frame carries no motion.
_RAIN = 0.1
_BASE = timedelta(hours=1)


def motion(before: np.ndarray, after: np.ndarray) -> tuple[int, int]:
    """Shift in cells (rows, columns) that best moves ``before`` onto ``after``."""
    a, b = np.nan_to_num(before), np.nan_to_num(after)
    if a.max() < _RAIN or b.max() < _RAIN:
        return 0, 0
    cross = np.fft.fft2(b) * np.conj(np.fft.fft2(a))
    size = np.abs(cross)
    cross = np.divide(cross, size, out=np.zeros_like(cross), where=size > 0)
    surface = np.fft.ifft2(cross).real
    row, col = np.unravel_index(int(np.argmax(surface)), surface.shape)
    rows, cols = surface.shape
    # Peaks past half the field are negative shifts.
    row = int(row) - rows if row > rows // 2 else int(row)
    col = int(col) - cols if col > cols // 2 else int(col)
    return row, col


def _shift(field: np.ndarray, rows: int, cols: int) -> np.ndarray:
    out = field.copy()
    n, m = field.shape
    if abs(rows) >= n or abs(cols) >= m:
        return out
    target = (
        slice(max(rows, 0), n + min(rows, 0)),
        slice(max(cols, 0), m + min(cols, 0)),
    )
    source = (
        slice(max(-rows, 0), n + min(-rows, 0)),
        slice(max(-cols, 0), m + min(-cols, 0)),
    )
    out[target] = field[source]
    return out


class Extrapolation:
    name: ClassVar[str] = "extrapolation"

    def fit(self, history: xr.DataArray) -> None:
        """Nothing to fit."""

    def predict(
        self,
        context: xr.DataArray,
        covariates: xr.DataArray | None,
        issue_time: datetime,
        leads: Sequence[timedelta],
    ) -> xr.DataArray:
        check_context(context, issue_time)
        if set(context.dims) != {"time", "latitude", "longitude"}:
            raise ValueError("extrapolation needs a gridded context")
        frames = context.transpose("time", "latitude", "longitude")
        last = frames.values[-1]
        earlier = frames.time.values == frames.time.values[-1] - np.timedelta64(_BASE)
        rows, cols = 0, 0
        if earlier.any():
            rows, cols = motion(frames.values[np.flatnonzero(earlier)[0]], last)
        moved = [
            _shift(last, round(rows * (lead / _BASE)), round(cols * (lead / _BASE)))
            for lead in leads
        ]
        out = xr.DataArray(
            np.stack(moved),
            dims=("lead", "latitude", "longitude"),
            coords={
                "lead": lead_coord(leads),
                "latitude": frames.latitude,
                "longitude": frames.longitude,
            },
        )
        return point_quantiles(out)
