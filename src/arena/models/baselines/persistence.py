"""Persistence: the last visible value, for every lead."""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import ClassVar

import xarray as xr

from arena.models.base import check_context, lead_coord, point_quantiles


class Persistence:
    name: ClassVar[str] = "persistence"

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
        last = context.isel(time=-1, drop=True)
        return point_quantiles(last.expand_dims(lead=lead_coord(leads)))
