from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from arena.config import load_config
from arena.data.sampling import split_of
from arena.eval.quantiles import LEVELS
from arena.eval.zero_shot import issue_times, model_parts
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
