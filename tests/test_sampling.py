from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr
from omegaconf import DictConfig

from arena.config import load_config
from arena.data.sampling import STRATA, split_of, strata, stratum_means


def transect(heights: list[float]) -> xr.DataArray:
    """One row of cells, 0.1 degree apart, running east."""
    lon = np.round(6.0 + 0.1 * np.arange(len(heights)), 6)
    return xr.DataArray(
        np.array([heights], dtype="float64"),
        dims=("latitude", "longitude"),
        coords={"latitude": [45.0], "longitude": lon},
    )


def names(labels: xr.DataArray) -> list[str]:
    return [STRATA[int(value)] for value in labels.values.ravel()]


def test_strata_along_a_transect_from_the_sea_to_the_mountains() -> None:
    labels = strata(transect([0, 0, 50, 200, 500, 1500]), load_config())
    assert names(labels) == ["sea", "sea", "coast", "coast", "pre_alps", "alps"]
    assert labels.dtype == np.int8
    assert labels.attrs["flag_meanings"] == "sea coast plains pre_alps alps"
    assert list(labels.attrs["flag_values"]) == [0, 1, 2, 3, 4]


def test_lowland_far_from_the_sea_is_plains() -> None:
    # 0.3 degrees still counts as coast, 0.4 degrees does not
    labels = strata(transect([0, 400, 400, 200, 200]), load_config())
    assert names(labels) == ["sea", "pre_alps", "pre_alps", "coast", "plains"]


def test_land_below_sea_level_is_not_sea() -> None:
    labels = strata(transect([0, -2, -2, -2, -2, -2]), load_config())
    assert names(labels) == ["sea", "coast", "coast", "coast", "plains", "plains"]


def test_thresholds_are_inclusive_at_the_lower_bound() -> None:
    labels = strata(transect([299.9, 300, 999.9, 1000]), load_config())
    assert names(labels) == ["plains", "pre_alps", "pre_alps", "alps"]


def test_stratum_means_average_the_cells_of_each_stratum() -> None:
    labels = strata(transect([0, 50, 400, 400, 2000]), load_config())
    field = xr.DataArray(
        np.array([[[1.0, 2.0, 3.0, 5.0, 7.0]], [[10.0, 20.0, 30.0, 50.0, 70.0]]]),
        dims=("time", "latitude", "longitude"),
        coords={"latitude": labels.latitude, "longitude": labels.longitude},
    )
    out = stratum_means(field, labels)
    # no cell is plains, so that stratum is left out
    assert list(out.stratum.values) == ["sea", "coast", "pre_alps", "alps"]
    assert out.dims == ("stratum", "time")
    np.testing.assert_allclose(out.sel(stratum="pre_alps").values, [4.0, 40.0])
    np.testing.assert_allclose(out.sel(stratum="alps").values, [7.0, 70.0])


def windows(cfg: DictConfig, family: str) -> dict[str, tuple[datetime, datetime]]:
    """Each window as [start, end) with the inclusive end date made exclusive."""
    splits = cfg.data.splits
    named = {
        "train": splits[family].train,
        "validation": splits[family].validation,
        "test": splits.test,
    }
    return {
        name: (
            datetime.fromisoformat(start),
            datetime.fromisoformat(end) + timedelta(days=1),
        )
        for name, (start, end) in named.items()
    }


@pytest.mark.parametrize("family", ["tsfm", "hybrid"])
def test_windows_of_a_family_are_ordered_and_disjoint(family: str) -> None:
    spans = list(windows(load_config(), family).values())
    for (_, end), (start, _) in zip(spans, spans[1:], strict=False):
        assert end <= start


def test_test_window_is_disjoint_from_every_fitting_window() -> None:
    cfg = load_config()
    test_start, test_end = windows(cfg, "tsfm")["test"]
    for family in ("tsfm", "hybrid"):
        for name in ("train", "validation"):
            start, end = windows(cfg, family)[name]
            assert end <= test_start or start >= test_end


def test_samples_within_the_embargo_of_a_window_end_are_dropped() -> None:
    cfg = load_config()
    end = datetime(2026, 11, 16)  # the day after the last test day
    assert split_of(end - timedelta(hours=48), cfg, "tsfm") == "test"
    assert split_of(end - timedelta(hours=47), cfg, "tsfm") is None
    assert split_of(datetime(2026, 5, 13), cfg, "hybrid") == "test"
    assert split_of(datetime(2026, 5, 12, 12), cfg, "hybrid") is None
    assert split_of(datetime(2015, 12, 31), cfg, "tsfm") is None


@pytest.mark.parametrize("family", ["tsfm", "hybrid"])
def test_no_target_reaches_outside_the_window_of_its_sample(family: str) -> None:
    cfg = load_config()
    spans = windows(cfg, family)
    embargo = timedelta(hours=int(cfg.data.splits.embargo_hours))
    issue = datetime(2016, 1, 1)
    counts = dict.fromkeys(spans, 0)
    while issue < datetime(2027, 1, 1):
        name = split_of(issue, cfg, family)
        if name is not None:
            start, end = spans[name]
            assert start <= issue and issue + embargo <= end
            counts[name] += 1
        issue += timedelta(hours=6)
    assert all(counts.values())
