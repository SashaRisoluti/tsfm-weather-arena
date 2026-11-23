"""The interface every model implements, and helpers shared by them."""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np
import xarray as xr

from arena.eval.quantiles import LEVELS


@runtime_checkable
class Forecaster(Protocol):
    """A model that turns what is visible at an issue time into quantiles.

    ``context`` holds the target up to the issue time, with a ``time``
    dimension and spatial dimensions (a grid, or a ``pixel`` dimension with
    latitude and longitude coordinates). ``covariates`` holds features from
    the feature providers, or is ``None``. ``predict`` returns the common
    quantile levels with dimensions ``(quantile, lead, *space)``.
    """

    name: ClassVar[str]

    def fit(self, history: xr.DataArray) -> None: ...

    def predict(
        self,
        context: xr.DataArray,
        covariates: xr.DataArray | None,
        issue_time: datetime,
        leads: Sequence[timedelta],
    ) -> xr.DataArray: ...


def check_context(context: xr.DataArray, issue_time: datetime) -> None:
    """Refuse a context holding target values from after the issue time."""
    if context.time.values[-1] > np.datetime64(issue_time):
        raise ValueError("context holds target values after the issue time")


def point_quantiles(values: xr.DataArray) -> xr.DataArray:
    """A point forecast as the common quantiles, all equal to the value."""
    return values.expand_dims(quantile=LEVELS).copy()


def lead_coord(leads: Sequence[timedelta]) -> list[np.timedelta64]:
    return [np.timedelta64(lead) for lead in leads]
