"""What a forecaster may see at an issue time, and on which time axis.

Every join here is "as of" an issue time: data enter only once they would
have been published. Totals over 6 hours end at the valid time, like the
accumulation windows of the forecast steps.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import TypeVar

import numpy as np
import xarray as xr
from omegaconf import DictConfig

from arena.data.aifs_archive import latest_available_init

T = TypeVar("T", xr.Dataset, xr.DataArray)


def six_hour_totals(rate: xr.DataArray, step_minutes: int) -> xr.DataArray:
    """Totals in mm over (T - 6 h, T] for T at 00, 06, 12 and 18 UTC.

    ``rate`` is in mm/h at ``step_minutes`` spacing; each frame stands for
    one step. A window with any frame missing is missing.
    """
    frames = 6 * 60 // step_minutes
    windows = rate.resample(time="6h", closed="right", label="right")
    out = windows.sum(min_count=frames) * (step_minutes / 60)
    out.attrs = {"long_name": "Precipitation over the previous 6 hours", "units": "mm"}
    return out


def past_as_of(ds: T, issue: datetime, lag_minutes: int) -> T:
    """Steps of ``ds`` published by ``issue``, given a publication lag."""
    latest = np.datetime64(issue - timedelta(minutes=lag_minutes))
    return ds.isel(time=np.flatnonzero(ds.time.values <= latest))


def nwp_as_of(
    runs: xr.Dataset, issue: datetime, leads_hours: Sequence[int], cfg: DictConfig
) -> xr.Dataset:
    """Forecast for ``leads_hours`` after ``issue`` from the newest usable run.

    ``runs`` holds per-step totals with dimensions ``init_time`` and ``step``.
    The result has a ``lead`` dimension and keeps the run start and the step
    used for each lead. Raises ``KeyError`` if that run or step is not cached.
    """
    init = latest_available_init(cfg, issue)
    if np.datetime64(init) not in runs.init_time.values:
        raise KeyError(f"run {init:%Y-%m-%d %Hz} is not cached")
    offset = int((issue - init) / timedelta(hours=1))
    steps = [offset + int(lead) for lead in leads_hours]
    missing = sorted(set(steps) - set(runs.step.values.tolist()))
    if missing:
        raise KeyError(f"run {init:%Y-%m-%d %Hz} has no steps {missing}")
    run = runs.sel(init_time=np.datetime64(init), step=steps)
    return run.rename(step="lead").assign_coords(
        lead=list(leads_hours),
        step=("lead", steps),
        issue_time=np.datetime64(issue, "ns"),
    )
