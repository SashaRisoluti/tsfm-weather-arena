"""Zero-shot runs: every model scored on the same issues, leads and pixels.

Scores are saved as parts, one file per model and issue day, under
``results/parts/<regime>/<label>/<model>/``. A run that stops can be started
again and skips the days already done; the summary is built from all parts.
"""

import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import DictConfig, OmegaConf

from arena.data import cache
from arena.data.align import past_as_of, six_hour_totals
from arena.data.sampling import split_of
from arena.eval.runner import _at, score_parts, summarise
from arena.finetune.windows import to_steps
from arena.models.base import Forecaster
from arena.models.baselines.climatology import Climatology
from arena.models.baselines.extrapolation import Extrapolation
from arena.models.baselines.persistence import Persistence
from arena.paths import project_path
from arena.physics.base import covariates

Covariates = Callable[[datetime], xr.DataArray]
log = logging.getLogger(__name__)
_STEP = {"a": timedelta(minutes=15), "b": timedelta(hours=6)}


def issue_times(cfg: DictConfig, regime: str, split: str) -> list[datetime]:
    """Issue times of a regime for ``smoke`` (validation days) or ``test``."""
    if split == "smoke":
        smoke = cfg.tsfm.smoke[f"regime_{regime}"]
        if regime == "a":
            every = int(smoke.issue_every_hours)
            days = [datetime.fromisoformat(d) for d in smoke.days]
            issues = [d + timedelta(hours=h) for d in days for h in range(0, 24, every)]
        else:
            issues = _daily(smoke.start, smoke.end, cfg.tsfm.regime_b.issue_hours)
        expected = "validation"
    elif split == "test":
        first, last = cfg.data.splits.test
        hours = range(24) if regime == "a" else cfg.tsfm.regime_b.issue_hours
        issues = _daily(first, last, hours)
        expected = "test"
    else:
        raise ValueError(f"unknown split {split!r}; expected smoke or test")
    return [i for i in issues if split_of(i, cfg, "tsfm") == expected]


def _daily(first: str, last: str, hours: Sequence[int]) -> list[datetime]:
    start, end = datetime.fromisoformat(first), datetime.fromisoformat(last)
    days = (end - start).days + 1
    return [start + timedelta(days=d, hours=int(h)) for d in range(days) for h in hours]


def model_parts(
    target: xr.DataArray,
    model: Forecaster,
    issues: Sequence[datetime],
    leads: Sequence[timedelta],
    context_steps: int,
    thresholds: Sequence[float],
    pixels: pd.DataFrame,
    covariates_for: Covariates | None = None,
    lag_minutes: int = 0,
) -> Iterator[tuple[date, pd.DataFrame]]:
    """Score parts of one model, one table per issue day.

    The context of an issue is the last ``context_steps`` of the target
    visible at the issue time, given ``lag_minutes``.
    """
    by_day: dict[date, list[datetime]] = defaultdict(list)
    for issue in issues:
        by_day[issue.date()].append(issue)
    lead_axis = [np.timedelta64(lead) for lead in leads]
    for day, day_issues in sorted(by_day.items()):
        forecasts, observed = [], []
        for issue in day_issues:
            visible = past_as_of(
                target.sel(time=slice(None, issue)), issue, lag_minutes
            )
            context = visible.isel(time=slice(-context_steps, None))
            features = None
            if covariates_for is not None and getattr(model, "use_covariates", False):
                features = covariates_for(issue)
            forecast = model.predict(context, features, issue, leads)
            forecasts.append(_at(forecast, pixels).reset_coords(drop=True))
            seen = target.sel(time=[issue + lead for lead in leads])
            seen = _at(seen, pixels).rename(time="lead").assign_coords(lead=lead_axis)
            observed.append(seen.reset_coords(drop=True))
        part = score_parts(
            xr.concat(forecasts, dim="issue"),
            xr.concat(observed, dim="issue"),
            thresholds,
        ).reset_index()
        part["lead_minutes"] = part.pop("lead") // pd.Timedelta(minutes=1)
        yield day, part


def satellite_covariates(cfg: DictConfig, start: datetime, end: datetime) -> Covariates:
    """Covariates on the 15-minute steps of each issue's context and horizon:
    satellite features as published (lag), solar features known ahead."""
    settings = cfg.tsfm.regime_a
    hourly = cache.open_cached(cfg, "satweave").sel(time=slice(start, end)).load()
    names = list(settings.covariates)
    past = covariates(hourly, [n for n in names if n != "solar"])
    lag = int(settings.satellite_lag_minutes)
    before, after = int(settings.context_steps), int(settings.horizon_steps)

    def build(issue: datetime) -> xr.DataArray:
        steps = pd.DatetimeIndex(
            [issue + _STEP["a"] * k for k in range(-before + 1, after + 1)]
        )
        parts = [to_steps(past, steps, lag)]
        if "solar" in names:
            grid = xr.Dataset(
                coords={
                    "time": steps,
                    "latitude": past.latitude,
                    "longitude": past.longitude,
                }
            )
            parts.append(covariates(grid, ["solar"]))
        return xr.concat(parts, dim="feature")

    return build


def _device(cfg: DictConfig) -> str:
    if cfg.tsfm.device != "auto":
        return str(cfg.tsfm.device)
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@contextmanager
def _backends(cfg: DictConfig) -> Iterator[Callable[[str], Any]]:
    """Backends loaded on first use and shared by the runs of one model."""
    loaded: dict[str, Any] = {}

    def get(model: str) -> Any:
        if model not in loaded:
            checkpoint = str(cfg.tsfm.checkpoints[model])
            if model == "timesfm3":
                from arena.models.tsfm.timesfm3 import TimesFM3Backend

                loaded[model] = TimesFM3Backend(
                    checkpoint,
                    device=_device(cfg),
                    batch_size=int(cfg.tsfm.timesfm3_batch_size),
                )
            else:
                from arena.models.tsfm.timestral import TimestralBackend

                loaded[model] = TimestralBackend(
                    checkpoint,
                    device=_device(cfg),
                    batch_size=int(cfg.tsfm.timestral_batch_size),
                )
        return loaded[model]

    try:
        yield get
    finally:
        for backend in loaded.values():
            if hasattr(backend, "close"):
                backend.close()


def run_zero_shot(
    cfg: DictConfig, regime: str, split: str, runs: Sequence[str] | None = None
) -> Path:
    """Score the baselines and the configured runs; return the summary file."""
    from arena.models.tsfm.base import TsfmForecaster

    settings = cfg.tsfm[f"regime_{regime}"]
    unknown = set(runs or []) - {str(r.name) for r in settings.runs}
    if unknown:
        raise ValueError(f"unknown runs for regime {regime}: {sorted(unknown)}")
    issues = issue_times(cfg, regime, split)
    if not issues:
        raise ValueError(f"no {split} issue times for regime {regime}")
    step = _STEP[regime]
    context_steps = int(settings.context_steps)
    if regime == "a":
        leads = [step * k for k in range(1, int(settings.horizon_steps) + 1)]
    else:
        leads = [timedelta(hours=int(h)) for h in cfg.data.align.regime_b.leads_hours]
    thresholds = list(cfg.eval.thresholds[f"regime_{regime}"])
    pixel_file = str(cfg.eval.pixels.file).format(region=cfg.region.name)
    pixels = pd.read_csv(project_path(pixel_file))
    shared = {
        "context_steps": context_steps,
        "leads_minutes": [lead // timedelta(minutes=1) for lead in leads],
        "thresholds": thresholds,
        "pixels": hashlib.sha256(project_path(pixel_file).read_bytes()).hexdigest(),
        "issue_hours": sorted({i.hour for i in issues}),
        "train": list(cfg.data.splits.tsfm.train),
    }

    rain = cache.open_cached(cfg, "pardig").precipitation
    start = issues[0] - step * (context_steps + 1) - timedelta(hours=6)
    end = issues[-1] + max(leads)
    target = rain.sel(time=slice(start, end)).load()
    train = cfg.data.splits.tsfm.train
    history = _at(rain, pixels).sel(time=slice(train[0], f"{train[1]}T23:59")).load()
    if regime == "b":
        target = six_hour_totals(target, int(cfg.data.pardig.step_minutes))
        history = six_hour_totals(history, int(cfg.data.pardig.step_minutes))
    climatology = Climatology()
    climatology.fit(history)

    models: dict[str, Any] = {"persistence": Persistence(), "climatology": climatology}
    if regime == "a":
        models["extrapolation"] = Extrapolation()
    chosen = [r for r in settings.runs if runs is None or r.name in runs]
    needs_covariates = any(r.get("covariates", False) for r in chosen)
    covariates_for = (
        satellite_covariates(cfg, start - timedelta(hours=2), end)
        if needs_covariates
        else None
    )
    folder = project_path(str(cfg.tsfm.parts_dir)) / regime / split
    with _backends(cfg) as backend:
        for run in [None, *chosen]:
            names = list(models) if run is None else [str(run.name)]
            for name in names:
                done = folder / name
                claim(done, {**shared, **_run_settings(cfg, regime, name, run)})
                todo = [i for i in issues if not (done / f"{i:%Y%m%d}.csv").exists()]
                if not todo:
                    continue
                model = models.get(name) or TsfmForecaster(
                    name,
                    backend(str(run.model)),
                    pixels,
                    context_steps,
                    neighbours=bool(run.get("neighbours", False)),
                    use_covariates=bool(run.get("covariates", False)),
                    transform=str(run.get("transform", "none")),
                )
                for day, part in model_parts(
                    target,
                    model,
                    todo,
                    leads,
                    context_steps,
                    thresholds,
                    pixels,
                    covariates_for,
                ):
                    part.insert(0, "model", name)
                    write_part(part, done / f"{day:%Y%m%d}.csv")
    names = [*models, *(str(r.name) for r in settings.runs)]
    days = sorted({i.date() for i in issues})
    return summarise_runs(cfg, regime, split, names, days)


def _run_settings(cfg: DictConfig, regime: str, name: str, run: Any) -> dict[str, Any]:
    if run is None:
        return {"model": name}
    settings = {
        "run": OmegaConf.to_container(run),
        "checkpoint": str(cfg.tsfm.checkpoints[run.model]),
    }
    if run.get("covariates", False):
        regime_a = cfg.tsfm.regime_a
        settings["satellite_lag_minutes"] = int(regime_a.satellite_lag_minutes)
        settings["covariates"] = list(regime_a.covariates)
    return settings


def claim(folder: Path, settings: dict[str, Any]) -> None:
    """Tie a folder of parts to the settings that produced them.

    Parts made with other settings would be summed silently with new ones,
    so a mismatch is refused.
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "settings.json"
    text = json.dumps(settings, sort_keys=True, indent=1)
    if not path.exists():
        path.write_text(text, encoding="utf-8")
    elif json.loads(path.read_text(encoding="utf-8")) != json.loads(text):
        raise ValueError(
            f"{folder} holds parts made with other settings; "
            "move it away to score again"
        )


def write_part(part: pd.DataFrame, path: Path) -> None:
    """Write a part under a temporary name first: a run cut off mid-write
    must not leave a file that resuming would take as done."""
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        part.to_csv(tmp, index=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def summarise_runs(
    cfg: DictConfig,
    regime: str,
    split: str,
    names: Sequence[str],
    days: Sequence[date],
) -> Path:
    """Summary per model and lead over the given issue days.

    A model is included only when every day is scored; all included models
    must have been scored on the same number of samples.
    """
    folder = project_path(str(cfg.tsfm.parts_dir)) / regime / split
    frames = []
    for name in names:
        files = [folder / name / f"{day:%Y%m%d}.csv" for day in days]
        done = sum(f.exists() for f in files)
        if done < len(files):
            if done:
                log.warning(
                    "%s: %d of %d days scored, left out", name, done, len(files)
                )
            continue
        frames += [pd.read_csv(f) for f in files]
    parts = pd.concat(frames)
    if parts.isna().any().any():
        raise ValueError("parts with missing scores: settings changed between days")
    samples = parts.groupby(["lead_minutes", "model"]).samples.sum().unstack()
    if (samples.nunique(axis=1) > 1).any():
        raise ValueError("models were scored on different numbers of samples")
    table = summarise(parts, by=("model", "lead_minutes")).reset_index()
    reference = table[table.model == "climatology"].set_index("lead_minutes").crps
    table["crps_skill"] = 1 - table.crps / table.lead_minutes.map(reference)
    path = project_path(str(cfg.tsfm.summary).format(regime=regime, label=split))
    path.parent.mkdir(parents=True, exist_ok=True)
    table.round(5).to_csv(path, index=False)
    return path
