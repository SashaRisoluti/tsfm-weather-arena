"""Stored AIFS ensemble forecasts, served as a forecaster."""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import ClassVar

import xarray as xr
from omegaconf import DictConfig

from arena.data.align import nwp_as_of
from arena.data.grid import common_grid, conservative_regrid
from arena.data.region import Region
from arena.eval.quantiles import from_ensemble
from arena.models.base import lead_coord


class AifsReference:
    """6-hour totals of the newest usable run, as quantiles of its members.

    ``runs`` holds per-step totals (see ``aifs_ens.deaccumulate``). Members
    are remapped to the common grid first; the output is on the points of
    the context, its grid cells or its pixels.
    """

    name: ClassVar[str] = "aifs_ens"

    def __init__(self, runs: xr.Dataset, cfg: DictConfig) -> None:
        self.runs = runs
        self.cfg = cfg

    def fit(self, history: xr.DataArray) -> None:
        """Nothing to fit."""

    def predict(
        self,
        context: xr.DataArray,
        covariates: xr.DataArray | None,
        issue_time: datetime,
        leads: Sequence[timedelta],
    ) -> xr.DataArray:
        if any(lead % timedelta(hours=6) for lead in leads):
            raise ValueError("forecast leads must be whole multiples of 6 h")
        hours = [lead // timedelta(hours=1) for lead in leads]
        forecast = nwp_as_of(self.runs[["tp"]], issue_time, hours, self.cfg)
        region = Region.from_config(self.cfg.region)
        lat, lon = common_grid(region, float(self.cfg.data.grid_step))
        fine = conservative_regrid(forecast.tp, lat, lon)
        q = from_ensemble(fine).assign_coords(lead=lead_coord(leads))
        q = q.drop_vars(["step", "init_time", "issue_time"], errors="ignore")
        # Grid coordinates or per-pixel coordinates select the same way.
        points = q.sel(latitude=context.latitude, longitude=context.longitude)
        return points.transpose("quantile", "lead", ...)
