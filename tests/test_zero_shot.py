from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from arena.config import load_config
from arena.data.sampling import split_of
from arena.eval.quantiles import LEVELS
from arena.eval.zero_shot import (
    claim,
    issue_times,
    model_parts,
    run_zero_shot,
    summarise_runs,
    write_part,
)
from arena.models.baselines.persistence import Persistence

CFG = load_config()


def test_smoke_issues_are_validation_days_at_the_configured_cadence() -> None:
    a = issue_times(CFG, "a", "smoke")
    assert len(a) == 3 * 8 and a[0] == datetime(2025, 7, 15, 0) and a[1].hour == 3
    b = issue_times(CFG, "b", "smoke")
    assert len(b) == 62 * 2 and {i.hour for i in b} == {0, 12}
    assert all(split_of(i, CFG, "tsfm") == "validation" for i in a + b)


def test_test_issues_stop_an_embargo_before_the_window_end() -> None:
    a = issue_times(CFG, "a", "test")
    assert a[0] == datetime(2026, 5, 13, 0)
    assert a[-1] == datetime(2026, 11, 14, 0)  # 48 h before the end of 15 November
    assert len({i.hour for i in a}) == 24
    with pytest.raises(ValueError):
        issue_times(CFG, "a", "validation")


def rain(hours: int = 48) -> xr.DataArray:
    times = pd.date_range("2025-07-15", periods=hours * 4, freq="15min")
    values = np.arange(times.size, dtype="float64")[:, None] * np.array([1.0, 2.0])
    return xr.DataArray(
        values,
        dims=("time", "pixel"),
        coords={
            "time": times,
            "latitude": ("pixel", [45.0, 46.0]),
            "longitude": ("pixel", [7.0, 8.0]),
        },
    )


PIXELS = pd.DataFrame({"latitude": [45.0, 46.0], "longitude": [7.0, 8.0]})


def test_parts_come_one_table_per_issue_day() -> None:
    issues = [
        datetime(2025, 7, 15, 12),
        datetime(2025, 7, 15, 18),
        datetime(2025, 7, 16, 6),
    ]
    leads = [timedelta(minutes=15), timedelta(hours=1)]
    parts = dict(model_parts(rain(), Persistence(), issues, leads, 8, [1.0], PIXELS))
    assert sorted(parts) == [datetime(2025, 7, 15).date(), datetime(2025, 7, 16).date()]
    first = parts[datetime(2025, 7, 15).date()].set_index("lead_minutes")
    assert first.loc[15, "samples"] == 4
    # persistence misses by one step (1 and 2 per pixel) at 15 min, four at 1 h
    assert first.loc[15, "crps_sum"] == pytest.approx(2 * (1 + 2))
    assert first.loc[60, "crps_sum"] == pytest.approx(2 * (4 + 8))


class CovariateModel:
    """Records what it is given and forecasts zero."""

    name = "spy"
    use_covariates = True

    def __init__(self) -> None:
        self.seen: list[tuple[datetime, pd.Timestamp, int]] = []

    def fit(self, history: xr.DataArray) -> None: ...

    def predict(self, context, covariates, issue_time, leads):  # type: ignore[no-untyped-def]
        self.seen.append(
            (issue_time, pd.Timestamp(context.time.values[-1]), int(covariates))
        )
        zero = xr.zeros_like(context.isel(time=0, drop=True))
        out = zero.expand_dims(lead=[np.timedelta64(lead) for lead in leads])
        return out.expand_dims(quantile=LEVELS).copy()


def test_covariates_are_built_per_issue_and_context_ends_at_the_visible_time() -> None:
    spy = CovariateModel()
    issues = [datetime(2025, 7, 15, 12), datetime(2025, 7, 15, 13)]
    built: list[datetime] = []

    def covariates_for(issue: datetime) -> xr.DataArray:
        built.append(issue)
        return xr.DataArray(issue.hour)

    list(
        model_parts(
            rain(),
            spy,
            issues,
            [timedelta(minutes=15)],
            4,
            [1.0],
            PIXELS,
            covariates_for,
            lag_minutes=90,
        )
    )
    assert built == issues
    assert [s[1] for s in spy.seen] == [
        pd.Timestamp("2025-07-15T10:30"),
        pd.Timestamp("2025-07-15T11:30"),
    ]
    assert [s[2] for s in spy.seen] == [12, 13]
    # a model without covariates never triggers the builder
    built.clear()
    list(
        model_parts(
            rain(),
            Persistence(),
            issues,
            [timedelta(minutes=15)],
            4,
            [1.0],
            PIXELS,
            covariates_for,
        )
    )
    assert built == []


DAYS = [datetime(2025, 7, 15).date(), datetime(2025, 7, 16).date()]


def write_parts(folder, model: str, days) -> None:  # type: ignore[no-untyped-def]
    issues = [
        datetime.combine(day, datetime.min.time()) + timedelta(hours=6) for day in days
    ]
    parts = model_parts(
        rain(96), Persistence(), issues, [timedelta(minutes=15)], 8, [1.0], PIXELS
    )
    (folder / model).mkdir(parents=True, exist_ok=True)
    for day, part in parts:
        part.insert(0, "model", model)
        part.to_csv(folder / model / f"{day:%Y%m%d}.csv", index=False)


def summary_config(tmp_path):  # type: ignore[no-untyped-def]
    cfg = load_config()
    cfg.tsfm.parts_dir = (tmp_path / "parts").as_posix()
    cfg.tsfm.summary = (tmp_path / "{regime}_{label}.csv").as_posix()
    return cfg, tmp_path / "parts" / "a" / "smoke"


def test_summary_reads_only_the_given_models_and_complete_days(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg, folder = summary_config(tmp_path)
    write_parts(folder, "climatology", DAYS)
    write_parts(folder, "persistence", DAYS)
    write_parts(folder, "removed_run", DAYS)
    write_parts(folder, "half_done", DAYS[:1])
    write_parts(folder, "persistence", [datetime(2025, 7, 17).date()])
    names = ["climatology", "persistence", "half_done"]
    table = pd.read_csv(summarise_runs(cfg, "a", "smoke", names, DAYS))
    assert sorted(table.model) == ["climatology", "persistence"]
    assert set(table.samples) == {2 * len(PIXELS)}


def test_summary_refuses_models_scored_on_different_samples(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg, folder = summary_config(tmp_path)
    write_parts(folder, "climatology", DAYS)
    write_parts(folder, "persistence", DAYS)
    path = folder / "persistence" / "20250716.csv"
    part = pd.read_csv(path)
    part["samples"] -= 1
    part.to_csv(path, index=False)
    with pytest.raises(ValueError, match="samples"):
        summarise_runs(cfg, "a", "smoke", ["climatology", "persistence"], DAYS)


def test_summary_refuses_parts_with_missing_scores(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg, folder = summary_config(tmp_path)
    write_parts(folder, "climatology", DAYS)
    path = folder / "climatology" / "20250716.csv"
    part = pd.read_csv(path)
    part["brier_2_sum"] = 0.0  # a threshold added after the first day was scored
    part.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing"):
        summarise_runs(cfg, "a", "smoke", ["climatology"], DAYS)


def test_parts_of_other_settings_are_refused(tmp_path) -> None:  # type: ignore[no-untyped-def]
    claim(tmp_path, {"context_steps": 672})
    claim(tmp_path, {"context_steps": 672})
    with pytest.raises(ValueError, match="settings"):
        claim(tmp_path, {"context_steps": 1344})


def test_a_part_cut_off_while_writing_leaves_no_file(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def cut_off(self, path, **kwargs):  # type: ignore[no-untyped-def]
        Path(path).write_text("model,samp")
        raise KeyboardInterrupt

    monkeypatch.setattr(pd.DataFrame, "to_csv", cut_off)
    with pytest.raises(KeyboardInterrupt):
        write_part(pd.DataFrame({"a": [1]}), tmp_path / "20250715.csv")
    assert list(tmp_path.iterdir()) == []


def test_unknown_run_names_are_refused() -> None:
    with pytest.raises(ValueError, match="timesfm_3"):
        run_zero_shot(CFG, "a", "smoke", ["timesfm_3"])
