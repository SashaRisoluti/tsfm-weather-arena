"""Climatology: quantiles of past values at the same place, month and hour."""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import ClassVar

import numpy as np
import pandas as pd
import xarray as xr

from arena.eval.quantiles import LEVELS
from arena.models.base import check_context, lead_coord


def _key(times: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(times.month * 100 + times.hour)


class Climatology:
    """Quantiles per location, calendar month and UTC hour of the valid time.

    Fitted on the training window only; the caller passes that window.
    """

    name: ClassVar[str] = "climatology"

    def __init__(self) -> None:
        self.table: xr.DataArray | None = None

    def fit(self, history: xr.DataArray) -> None:
        key = xr.DataArray(
            _key(pd.DatetimeIndex(history.time.values)), dims="time", name="key"
        )
        table = history.groupby(key).quantile(LEVELS, dim="time")
        self.table = table.transpose("key", "quantile", ...)

    def predict(
        self,
        context: xr.DataArray,
        covariates: xr.DataArray | None,
        issue_time: datetime,
        leads: Sequence[timedelta],
    ) -> xr.DataArray:
        if self.table is None:
            raise RuntimeError("climatology is not fitted")
        check_context(context, issue_time)
        valid = pd.DatetimeIndex([issue_time + lead for lead in leads])
        rows = self.table.sel(key=_key(valid))  # KeyError for an unseen month-hour
        out = rows.rename(key="lead").assign_coords(lead=lead_coord(leads))
        return out.transpose("quantile", "lead", ...)
