"""One forecaster for every time-series foundation model.

The model-specific part is a backend with a single method::

    forecast(context, past, known, horizon) -> (quantiles, levels)

``context`` is (series, variate, time); ``past`` (series, covariate, time)
and ``known`` (series, covariate, time + horizon) are optional; the result
is (series, variate, horizon, level) on the backend's own quantile levels.
Everything else (pixels, neighbours, covariates, transform, mapping to the
common levels) is done here, identically for every backend.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Protocol

import numpy as np
import pandas as pd
import xarray as xr

from arena.eval.quantiles import harmonise
from arena.finetune.windows import inverse, pixel_series, transform
from arena.models.base import check_context, lead_coord


class Backend(Protocol):
    def forecast(
        self,
        context: np.ndarray,
        past: np.ndarray | None,
        known: np.ndarray | None,
        horizon: int,
    ) -> tuple[np.ndarray, Sequence[float]]: ...


class TsfmForecaster:
    """Zero-shot forecasts of a foundation model at fixed pixels.

    ``context`` passed to ``predict`` is a grid (time, latitude, longitude)
    of the target; ``covariates`` a grid (feature, time, latitude, longitude)
    with a ``known_future`` flag per feature, covering the context and the
    horizon. Only features known ahead are read after the issue time.
    """

    name = "tsfm"

    def __init__(
        self,
        name: str,
        backend: Backend,
        pixels: pd.DataFrame,
        context_steps: int,
        neighbours: bool = False,
        use_covariates: bool = False,
        transform: str = "none",
    ) -> None:
        self.name = name  # type: ignore[misc]
        self.backend = backend
        self.pixels = pixels.reset_index(drop=True)
        self.context_steps = context_steps
        self.neighbours = neighbours
        self.use_covariates = use_covariates
        self.transform = transform

    def fit(self, history: xr.DataArray) -> None:
        """Zero-shot: nothing to fit."""

    def predict(
        self,
        context: xr.DataArray,
        covariates: xr.DataArray | None,
        issue_time: datetime,
        leads: Sequence[timedelta],
    ) -> xr.DataArray:
        check_context(context, issue_time)
        if context.sizes["time"] < self.context_steps:
            raise ValueError(f"model needs {self.context_steps} steps of context")
        window = context.isel(time=slice(-self.context_steps, None))
        step = pd.Timedelta(window.time.values[-1] - window.time.values[-2])
        offsets = [pd.Timedelta(lead) / step for lead in leads]
        if any(o != int(o) or o < 1 for o in offsets):
            raise ValueError("leads must be whole numbers of time steps")
        horizon = int(max(offsets))

        series = transform(
            pixel_series(window, self.pixels, self.neighbours), self.transform
        )
        past = known = None
        if self.use_covariates:
            if covariates is None:
                raise ValueError(f"{self.name} needs covariates")
            past, known = self._covariates(
                covariates, window.time.values, step, horizon
            )
        quantiles, levels = self.backend.forecast(
            series.astype("float32"), past, known, horizon
        )
        centre = inverse(np.asarray(quantiles)[:, 0], self.transform)
        picked = centre[:, [int(o) - 1 for o in offsets], :]
        out = xr.DataArray(
            picked,
            dims=("pixel", "lead", "quantile"),
            coords={
                "quantile": list(levels),
                "lead": lead_coord(leads),
                "latitude": ("pixel", self.pixels.latitude.to_numpy()),
                "longitude": ("pixel", self.pixels.longitude.to_numpy()),
            },
        )
        return harmonise(out).transpose("quantile", "lead", "pixel")

    def _covariates(
        self,
        features: xr.DataArray,
        context_times: np.ndarray,
        step: pd.Timedelta,
        horizon: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        future = context_times[-1] + step * np.arange(1, horizon + 1)
        flags = features.known_future.values
        past = known = None
        if (~flags).any():
            seen = features.isel(feature=np.flatnonzero(~flags)).sel(time=context_times)
            past = pixel_series(seen, self.pixels, neighbours=False)[:, 0]
        if flags.any():
            ahead = features.isel(feature=np.flatnonzero(flags))
            ahead = ahead.sel(time=np.concatenate([context_times, future]))
            known = pixel_series(ahead, self.pixels, neighbours=False)[:, 0]
        cast = [None if a is None else a.astype("float32") for a in (past, known)]
        return cast[0], cast[1]
