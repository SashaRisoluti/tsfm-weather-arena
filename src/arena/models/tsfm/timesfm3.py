"""TimesFM 3 (Google Research), run in-process.

Weights ``google/timesfm-3.0-pytorch`` are under the TimesFM Non-Commercial
License; the package ``timesfm`` is Apache 2.0. The model forecasts the
median and the levels 0.1 to 0.9, and accepts covariates known over the
context only or over the context and the horizon.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np


class TimesFM3Backend:
    def __init__(
        self,
        checkpoint: str = "google/timesfm-3.0-pytorch",
        device: str | None = None,
        model: Any = None,
    ) -> None:
        if model is None:
            # Imported here: torch and timesfm are an optional dependency group.
            from timesfm3 import TimesFM3Forecaster

            model = TimesFM3Forecaster.from_pretrained(checkpoint, device=device)
        self.model = model

    def forecast(
        self,
        context: np.ndarray,
        past: np.ndarray | None,
        known: np.ndarray | None,
        horizon: int,
    ) -> tuple[np.ndarray, Sequence[float]]:
        univariate = context.shape[1] == 1
        contexts = [series[0] if univariate else series for series in context]
        outputs = self.model.predict_batch(
            contexts,
            horizon=horizon,
            past_only_covariates=None if past is None else list(past),
            past_future_covariates=None if known is None else list(known),
            return_quantiles=True,
            make_positive=True,
        )
        quantiles = np.stack([np.asarray(out.quantiles) for out in outputs])
        if univariate:
            quantiles = quantiles[:, None]
        return quantiles, list(self.model.config.quantiles)
