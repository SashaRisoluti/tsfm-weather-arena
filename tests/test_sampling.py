import numpy as np
import xarray as xr

from arena.config import load_config
from arena.data.sampling import STRATA, strata


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
