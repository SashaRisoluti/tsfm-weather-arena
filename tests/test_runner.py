from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from arena.eval.quantiles import LEVELS
from arena.eval.runner import run_regime_a, score, score_parts, summarise
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


def test_score_counts_only_samples_with_a_forecast_and_an_observation() -> None:
    forecast = point(np.array([[2.0, 0.0, np.nan], [0.0, 6.0, 1.0]]))
    forecast = forecast.assign_coords(pixel=[0, 1, 2])
    observed = xr.DataArray(
        [[0.0, np.nan, 5.0], [3.0, 6.0, np.nan]],
        dims=("lead", "pixel"),
        coords={"lead": [1, 2], "pixel": [0, 1, 2]},
    )
    table = score(forecast, observed, thresholds=[1.0])
    # lead 1 keeps pixel 0 only, lead 2 pixels 0 and 1
    assert table["samples"].tolist() == [1, 2]
    np.testing.assert_allclose(table["crps"], [2.0, 1.5])
    np.testing.assert_allclose(table["coverage_90"], [0.0, 0.5])
    # lead 1: a false alarm only; lead 2: one miss, one hit
    np.testing.assert_allclose(table["far_1"], [1.0, 0.0])
    np.testing.assert_allclose(table["pod_1"], [np.nan, 0.5])
    # point forecasts: event probability 0.95 or 0.05
    np.testing.assert_allclose(table["brier_1"], [0.95**2, (0.95**2 + 0.05**2) / 2])


def test_score_sums_counts_before_taking_ratios() -> None:
    # per-sample ratios would average 0.5 and 0; summed counts give 1/3
    forecast = point(np.array([[5.0, 5.0], [0.0, 0.0]]))
    observed = xr.DataArray(
        [[5.0, 0.0], [0.0, 0.0]], dims=("lead", "pixel"), coords={"lead": [1, 2]}
    )
    forecast = xr.concat([forecast, forecast.isel(pixel=[1])], dim="pixel")
    observed = xr.concat([observed, observed.isel(pixel=[1])], dim="pixel")
    table = score(forecast, observed, thresholds=[1.0])
    assert table["csi_1"].iloc[0] == pytest.approx(1 / 3)


def test_score_refuses_quantiles_off_the_common_levels() -> None:
    forecast = point(np.zeros((2, 2))).isel(quantile=slice(1, None))
    observed = xr.DataArray(
        np.zeros((2, 2)), dims=("lead", "pixel"), coords={"lead": [1, 2]}
    )
    with pytest.raises(ValueError, match="common quantile levels"):
        score(forecast, observed, thresholds=[1.0])


def test_run_regime_a_can_cut_the_context_before_the_issue_time() -> None:
    times = pd.date_range("2025-07-25", periods=48, freq="15min")
    rain = xr.DataArray(
        np.arange(48.0)[:, None], dims=("time", "pixel"), coords={"time": times}
    )
    issue = datetime(2025, 7, 25, 6)
    table = run_regime_a(
        rain,
        {"persistence": Persistence()},
        [issue],
        [timedelta(minutes=15)],
        context=timedelta(hours=3),
        thresholds=[1.0],
        lag_minutes=90,
    )
    # last visible frame 04:30 holds 18; the observation at 06:15 holds 25
    assert table.crps.iloc[0] == pytest.approx(7.0)


def test_parts_of_two_batches_add_up_to_the_scores_of_both() -> None:
    rng = np.random.default_rng(5)
    values = rng.gamma(0.5, 2.0, (2, 6))
    forecast = point(values).copy(data=np.sort(rng.gamma(0.5, 2.0, (19, 2, 6)), axis=0))
    observed = xr.DataArray(values, dims=("lead", "pixel"), coords={"lead": [1, 2]})
    whole = score(forecast, observed, thresholds=[1.0])
    halves = pd.concat(
        [
            score_parts(forecast.isel(pixel=part), observed.isel(pixel=part), [1.0])
            for part in (slice(0, 3), slice(3, 6))
        ]
    ).reset_index()
    pd.testing.assert_frame_equal(summarise(halves), whole)
