from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from arena.eval.quantiles import LEVELS
from arena.eval.runner import run_regime_a, score
from arena.models.baselines.climatology import Climatology
from arena.models.baselines.persistence import Persistence


def point(values: np.ndarray) -> xr.DataArray:
    """Point forecasts with dimensions (quantile, lead, pixel)."""
    da = xr.DataArray(values, dims=("lead", "pixel"), coords={"lead": [1, 2]})
    return da.expand_dims(quantile=LEVELS).copy()


def test_score_by_hand() -> None:
    forecast = point(np.array([[2.0, 0.0], [0.0, 6.0]]))
    observed = xr.DataArray(
        [[0.0, 0.0], [3.0, 6.0]], dims=("lead", "pixel"), coords={"lead": [1, 2]}
    )
    table = score(forecast, observed, thresholds=[1.0])
    # point forecasts: CRPS is the absolute error
    np.testing.assert_allclose(table["crps"], [1.0, 1.5])
    np.testing.assert_allclose(table["mae_median"], [1.0, 1.5])
    # lead 1: one false alarm; lead 2: one hit, one miss
    np.testing.assert_allclose(table["csi_1"], [0.0, 0.5])
    np.testing.assert_allclose(table["coverage_90"], [0.5, 0.5])
    assert table["samples"].tolist() == [2, 2]


def test_run_regime_a_scores_each_model_and_lead() -> None:
    times = pd.date_range("2025-07-01", "2025-07-31T23:45", freq="15min")
    rng = np.random.default_rng(0)
    rain = xr.DataArray(
        rng.gamma(0.3, 1.0, (times.size, 2)),
        dims=("time", "pixel"),
        coords={
            "time": times,
            "latitude": ("pixel", [45.0, 46.0]),
            "longitude": ("pixel", [7.0, 8.0]),
        },
    )
    climatology = Climatology()
    climatology.fit(rain.sel(time=slice("2025-07-01", "2025-07-20")))
    issues = [datetime(2025, 7, 25, h) for h in (6, 12)]
    leads = [timedelta(minutes=15), timedelta(hours=1)]
    table = run_regime_a(
        rain,
        {"persistence": Persistence(), "climatology": climatology},
        issues,
        leads,
        context=timedelta(hours=1),
        thresholds=[1.0],
    )
    assert sorted(set(table.model)) == ["climatology", "persistence"]
    assert table.groupby("model").size().to_dict() == {
        "climatology": 2,
        "persistence": 2,
    }
    assert (table.samples == 4).all()
    row = table[(table.model == "persistence") & (table.lead_minutes == 15)].iloc[0]
    expected = np.mean(
        [
            abs(
                rain.sel(time=t).values
                - rain.sel(time=t + timedelta(minutes=15)).values
            )
            for t in issues
        ]
    )
    assert row.crps == pytest.approx(expected)
    skill = table[table.model == "climatology"].crps_skill
    np.testing.assert_allclose(skill, 0.0)
