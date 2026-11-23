"""Scoring loops: forecasts against observations, per model and lead."""

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import DictConfig

from arena.data import cache
from arena.data.align import past_as_of
from arena.data.sampling import split_of
from arena.eval import metrics
from arena.models.base import Forecaster
from arena.models.baselines.climatology import Climatology
from arena.models.baselines.extrapolation import Extrapolation
from arena.models.baselines.persistence import Persistence
from arena.paths import project_path


def score(
    forecast: xr.DataArray, observed: xr.DataArray, thresholds: Sequence[float]
) -> pd.DataFrame:
    """Scores per lead, averaged over every other dimension.

    ``forecast`` has dimensions ``(quantile, lead, ...)`` and ``observed``
    the same without ``quantile``. Categorical scores come from counts
    summed over the samples, with an event forecast where the median
    reaches the threshold.
    """
    samples = [str(dim) for dim in observed.dims if dim != "lead"]
    columns = {
        "crps": metrics.crps_quantile(forecast, observed).mean(samples),
        "mae_median": metrics.mae_median(forecast, observed).mean(samples),
        "samples": observed.notnull().sum(samples),
    }
    inside = metrics.coverage(forecast, observed)
    for width in inside.interval.values:
        columns[f"coverage_{round(width * 100)}"] = (
            inside.sel(interval=width, drop=True).astype("float64").mean(samples)
        )
    median = forecast.sel(quantile=0.5, drop=True)
    for threshold in thresholds:
        event = observed >= threshold
        p = metrics.exceedance_probability(forecast, threshold)
        columns[f"brier_{threshold:g}"] = metrics.brier(p, event).mean(samples)
        counts = metrics.categorical(median >= threshold, event, samples)
        for name in ("csi", "pod", "far"):
            columns[f"{name}_{threshold:g}"] = counts[name]
    table = xr.Dataset(columns).to_dataframe()
    return table.drop(columns=[c for c in table.columns if c not in columns])


def _at(forecast: xr.DataArray, pixels: pd.DataFrame | None) -> xr.DataArray:
    """The forecast at the evaluation pixels, if it is on a grid."""
    if pixels is None or "latitude" not in forecast.dims:
        return forecast
    return forecast.sel(
        latitude=xr.DataArray(pixels.latitude.to_numpy(), dims="pixel"),
        longitude=xr.DataArray(pixels.longitude.to_numpy(), dims="pixel"),
    )


def run_regime_a(
    rain: xr.DataArray,
    models: Mapping[str, Forecaster],
    issues: Sequence[datetime],
    leads: Sequence[timedelta],
    context: timedelta,
    thresholds: Sequence[float],
    pixels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score fitted models on the 15-minute target.

    The context of each issue is the target from ``issue - context`` up to
    the issue time (hindcast). If ``pixels`` is given, gridded forecasts and
    observations are scored at those pixels only. CRPS skill is relative to
    the model named ``climatology``, when it is among the models.
    """
    observed = xr.concat(
        [
            _at(rain.sel(time=[issue + lead for lead in leads]), pixels)
            .rename(time="lead")
            .assign_coords(lead=[np.timedelta64(lead) for lead in leads])
            for issue in issues
        ],
        dim="issue",
    )
    tables = []
    for name, model in models.items():
        forecasts = []
        for issue in issues:
            window = rain.sel(time=slice(issue - context, issue))
            seen = past_as_of(window, issue, 0)
            forecasts.append(_at(model.predict(seen, None, issue, leads), pixels))
        forecast = xr.concat(forecasts, dim="issue")
        table = score(forecast, observed, thresholds).reset_index()
        table.insert(0, "model", name)
        tables.append(table)
    out = pd.concat(tables, ignore_index=True)
    out.insert(1, "lead_minutes", out.pop("lead") // pd.Timedelta(minutes=1))
    if "climatology" in models:
        reference = out[out.model == "climatology"].set_index("lead_minutes").crps
        out["crps_skill"] = 1 - out.crps / out.lead_minutes.map(reference)
    return out


def smoke_regime_a(cfg: DictConfig) -> Path:
    """Baselines on a few validation days, written to a small CSV."""
    settings = cfg.eval.smoke
    days = [datetime.fromisoformat(day) for day in settings.days]
    issues = [day + timedelta(hours=h) for day in days for h in range(24)]
    wrong = [i for i in issues if split_of(i, cfg, "tsfm") != "validation"]
    if wrong:
        raise ValueError(f"smoke issues outside the validation window: {wrong[0]}")
    leads = [timedelta(minutes=int(m)) for m in settings.leads_minutes]
    context = timedelta(hours=int(settings.context_hours))
    pixels_file = project_path(str(cfg.eval.pixels.file).format(region=cfg.region.name))
    pixels = pd.read_csv(pixels_file)

    rain = cache.open_cached(cfg, "pardig").precipitation
    at_pixels = _at(rain, pixels)
    train = cfg.data.splits.tsfm.train
    climatology = Climatology()
    climatology.fit(at_pixels.sel(time=slice(train[0], f"{train[1]}T23:59")).load())
    span = rain.sel(time=slice(issues[0] - context, issues[-1] + max(leads))).load()
    table = run_regime_a(
        span,
        {
            "persistence": Persistence(),
            "climatology": climatology,
            "extrapolation": Extrapolation(),
        },
        issues,
        leads,
        context,
        list(cfg.eval.thresholds.regime_a),
        pixels,
    )
    path = project_path(str(settings.output))
    path.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(path, index=False)
    return path
