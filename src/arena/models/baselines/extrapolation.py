"""Lagrangian persistence: the last frame moved along the recent motion.

One motion vector per issue time: the shift, no faster than 120 km/h, that
maximises the cross-correlation of the frames one hour and zero hours
before the issue. The last frame is shifted
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
# Faster shifts are not considered: they come from new cells or other systems.
_MAX_KM_PER_HOUR = 120.0
_KM_PER_DEGREE = 111.2


def motion(
    before: np.ndarray,
    after: np.ndarray,
    max_rows: float = np.inf,
    max_cols: float = np.inf,
) -> tuple[int, int]:
    """Shift in cells (rows, columns) that best moves ``before`` onto ``after``.

    The shift maximises the cross-correlation of the two fields among shifts
    inside the ellipse with half-axes ``max_rows`` and ``max_cols``; (0, 0)
    if no allowed shift overlaps any rain.
    """
    a, b = np.nan_to_num(before), np.nan_to_num(after)
    if a.max() < _RAIN or b.max() < _RAIN:
        return 0, 0
    surface = np.fft.ifft2(np.fft.fft2(b) * np.conj(np.fft.fft2(a))).real
    rows, cols = surface.shape
    # Signed shift of every entry: past half the field it is negative.
    dr = np.fft.fftfreq(rows, 1 / rows)[:, None]
    dc = np.fft.fftfreq(cols, 1 / cols)[None, :]
    allowed = (dr / max_rows) ** 2 + (dc / max_cols) ** 2 <= 1
    surface = np.where(allowed, surface, -np.inf)
    peak = np.unravel_index(int(np.argmax(surface)), surface.shape)
    if surface[peak] <= 1e-9 * float((a * a).sum() + (b * b).sum()):
        return 0, 0
    return int(dr[peak[0], 0]), int(dc[0, peak[1]])


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
            lat = frames.latitude.values
            km_north = float(np.diff(lat).mean()) * _KM_PER_DEGREE
            km_east = (
                float(np.diff(frames.longitude.values).mean())
                * _KM_PER_DEGREE
                * np.cos(np.radians(lat.mean()))
            )
            before = frames.values[np.flatnonzero(earlier)[0]]
            rows, cols = motion(
                before,
                last,
                max_rows=_MAX_KM_PER_HOUR / km_north,
                max_cols=_MAX_KM_PER_HOUR / km_east,
            )
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
