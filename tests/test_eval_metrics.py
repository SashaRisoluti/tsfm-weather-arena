import numpy as np
import pytest
import scoringrules as sr
import xarray as xr

from arena.eval import metrics
from arena.eval.quantiles import LEVELS, from_ensemble, harmonise


def quantiles(values: list[float], levels: list[float]) -> xr.DataArray:
    return xr.DataArray(values, dims="quantile", coords={"quantile": levels})


def linear() -> xr.DataArray:
    """The common levels with q = 10 tau: 0.5, 1.0, ..., 9.5."""
    return quantiles(list(10 * LEVELS), list(LEVELS))


def test_common_levels() -> None:
    assert LEVELS.size == 19
    assert (LEVELS[0], LEVELS[9], LEVELS[-1]) == (0.05, 0.5, 0.95)


def test_harmonise_interpolates_and_extrapolates_by_hand() -> None:
    out = harmonise(quantiles([1.0, 3.0, 7.0], [0.1, 0.5, 0.9]))
    got = dict(zip(out["quantile"].values.round(2), out.values, strict=True))
    # slope 5 per unit probability below the median, 10 above
    assert got[0.05] == pytest.approx(0.75)
    assert got[0.3] == pytest.approx(2.0)
    assert got[0.7] == pytest.approx(5.0)
    assert got[0.95] == pytest.approx(7.5)


def test_harmonise_clips_at_zero_and_keeps_order() -> None:
    out = harmonise(quantiles([0.1, 1.0, 2.0], [0.1, 0.5, 0.9]))
    assert float(out.sel(quantile=0.05)) == 0.0
    assert bool((out.diff("quantile") >= 0).all())


def test_harmonise_keeps_other_dimensions() -> None:
    q = xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 4.0]]),
        dims=("pixel", "quantile"),
        coords={"quantile": [0.1, 0.5, 0.9]},
    )
    out = harmonise(q)
    assert out.dims == ("quantile", "pixel")
    np.testing.assert_allclose(out.sel(quantile=0.5).values, [2.0, 0.0])


def test_ensemble_to_quantiles_by_hand() -> None:
    members = xr.DataArray(np.arange(21.0), dims="member")
    out = from_ensemble(members)
    np.testing.assert_allclose(out.sel(quantile=0.05), 1.0)
    np.testing.assert_allclose(out.sel(quantile=0.5), 10.0)
    np.testing.assert_allclose(out["quantile"].values, LEVELS)


def test_crps_of_a_point_forecast_is_the_absolute_error() -> None:
    point = quantiles([2.0] * 19, list(LEVELS))
    assert float(metrics.crps_quantile(point, xr.DataArray(0.0))) == pytest.approx(2.0)


def test_crps_by_hand_on_three_levels() -> None:
    # pinball losses 0.125, 0 and 0.125; two thirds of their sum
    q = quantiles([0.0, 0.5, 1.0], [0.25, 0.5, 0.75])
    assert float(metrics.crps_quantile(q, xr.DataArray(0.5))) == pytest.approx(1 / 6)


def test_crps_matches_the_reference_implementation() -> None:
    rng = np.random.default_rng(0)
    obs = rng.gamma(0.5, 2.0, 200)
    fct = np.sort(rng.gamma(0.5, 2.0, (200, 19)), axis=1)
    q = xr.DataArray(fct, dims=("sample", "quantile"), coords={"quantile": LEVELS})
    ours = metrics.crps_quantile(q, xr.DataArray(obs, dims="sample"))
    np.testing.assert_allclose(
        ours.values, sr.crps_quantile(obs, fct, LEVELS), rtol=1e-12
    )

    ens = rng.gamma(0.5, 2.0, (200, 51))
    members = xr.DataArray(ens, dims=("sample", "member"))
    exact = metrics.crps_ensemble(members, xr.DataArray(obs, dims="sample"))
    np.testing.assert_allclose(
        exact.values, sr.crps_ensemble(obs, ens, estimator="nrg"), rtol=1e-10
    )


def test_ensemble_crps_by_hand() -> None:
    members = xr.DataArray([0.0, 1.0], dims="member")
    # mean |X - y| = 0.5, half the mean |X - X'| = 0.25
    assert float(metrics.crps_ensemble(members, xr.DataArray(0.5))) == pytest.approx(
        0.25
    )


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(3.0, 0.7), (6.25, 0.375), (0.2, 0.95), (20.0, 0.05)],
)
def test_exceedance_probability_from_quantiles(
    threshold: float, expected: float
) -> None:
    p = metrics.exceedance_probability(linear(), threshold)
    assert float(p) == pytest.approx(expected)


def test_exceedance_probability_of_a_point_forecast_has_flat_tails() -> None:
    point = quantiles([2.0] * 19, list(LEVELS))
    assert float(metrics.exceedance_probability(point, 1.0)) == pytest.approx(0.95)
    assert float(metrics.exceedance_probability(point, 3.0)) == pytest.approx(0.05)


def test_categorical_scores_by_hand() -> None:
    forecast = xr.DataArray([True, True, False, False, True], dims="sample")
    observed = xr.DataArray([True, False, True, False, True], dims="sample")
    scores = metrics.categorical(forecast, observed, dim="sample")
    # 2 hits, 1 false alarm, 1 miss
    assert float(scores.csi) == pytest.approx(0.5)
    assert float(scores.pod) == pytest.approx(2 / 3)
    assert float(scores.far) == pytest.approx(1 / 3)


def test_brier_by_hand() -> None:
    p = xr.DataArray([0.7, 0.2], dims="sample")
    o = xr.DataArray([True, False], dims="sample")
    np.testing.assert_allclose(metrics.brier(p, o).values, [0.09, 0.04])


def test_coverage_by_hand() -> None:
    q = linear()
    inside = metrics.coverage(q, xr.DataArray(3.0))
    assert bool(inside.sel(interval=0.5)) and bool(inside.sel(interval=0.9))
    inside = metrics.coverage(q, xr.DataArray(1.0))
    assert not bool(inside.sel(interval=0.5)) and bool(inside.sel(interval=0.9))


def test_pit_inside_the_distribution() -> None:
    rng = np.random.default_rng(0)
    assert float(metrics.pit(linear(), xr.DataArray(3.0), rng)) == pytest.approx(0.3)


def test_pit_of_a_dry_observation_is_randomised_over_the_mass_at_zero() -> None:
    rng = np.random.default_rng(0)
    # levels up to 0.3 are exactly zero: F(0) = 0.3
    values = [0.0 if level <= 0.3 else level for level in LEVELS]
    q = quantiles(values, list(LEVELS)).expand_dims(sample=4000)
    got = metrics.pit(q, xr.DataArray(np.zeros(4000), dims="sample"), rng)
    assert float(got.min()) >= 0.0 and float(got.max()) <= 0.3
    assert float(got.mean()) == pytest.approx(0.15, abs=0.01)
    # no level at zero: the flat lower tail gives F(0) = 0.05
    wet = metrics.pit(
        linear().expand_dims(sample=500),
        xr.DataArray(np.zeros(500), dims="sample"),
        rng,
    )
    assert float(wet.max()) <= 0.05


def test_reliability_by_hand() -> None:
    p = xr.DataArray([0.05, 0.05, 0.95, 0.95], dims="sample")
    o = xr.DataArray([False, True, True, True], dims="sample")
    table = metrics.reliability(p, o, bins=10)
    assert table.loc[0, "count"] == 2 and table.loc[0, "observed"] == 0.5
    assert table.loc[9, "count"] == 2 and table.loc[9, "observed"] == 1.0
    assert table.loc[0, "forecast"] == pytest.approx(0.05)


def field(cells: list[tuple[int, int]]) -> xr.DataArray:
    values = np.zeros((7, 7))
    for row, col in cells:
        values[row, col] = 5.0
    return xr.DataArray(values, dims=("latitude", "longitude"))


def test_fss_by_hand() -> None:
    obs, fc = field([(3, 3)]), field([(3, 4)])
    assert metrics.fss(obs, obs, threshold=1.0, window=3) == pytest.approx(1.0)
    assert metrics.fss(fc, obs, threshold=1.0, window=1) == pytest.approx(0.0)
    # 3 x 3 windows: six cells see both events (1/9 each), three see only one each
    assert metrics.fss(fc, obs, threshold=1.0, window=3) == pytest.approx(2 / 3)


def test_mae_of_the_median_by_hand() -> None:
    assert float(metrics.mae_median(linear(), xr.DataArray(2.0))) == pytest.approx(3.0)
