import numpy as np
import pytest
import xarray as xr

from arena.data.grid import cell_areas, common_grid, conservative_regrid
from arena.data.region import Region


def field(values: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
    )


# A 1 x 1 degree block: four 0.25 degree cells or ten 0.1 degree cells a side.
COARSE = np.round(0.125 + 0.25 * np.arange(4), 6)
FINE = np.round(0.05 + 0.1 * np.arange(10), 6)


def test_common_grid_spans_the_region_inclusive() -> None:
    lat, lon = common_grid(Region("alps", 6.0, 14.0, 43.5, 47.5), 0.1)
    assert (lat.size, lon.size) == (41, 81)
    assert (lat[0], lat[-1], lon[0], lon[-1]) == (43.5, 47.5, 6.0, 14.0)


def test_constant_field_stays_constant() -> None:
    src = field(np.full((4, 4), 3.5), 44 + COARSE, COARSE)
    out = conservative_regrid(src, 44 + FINE, FINE)
    np.testing.assert_allclose(out.values, 3.5, rtol=1e-12)


def test_mass_is_conserved_over_a_shared_domain() -> None:
    rng = np.random.default_rng(0)
    src = field(rng.gamma(0.5, 4.0, (4, 4)), 44 + COARSE, COARSE)
    out = conservative_regrid(src, 44 + FINE, FINE)
    before = float((src * cell_areas(src)).sum())
    after = float((out * cell_areas(out)).sum())
    assert after == pytest.approx(before, rel=1e-12)


def test_a_fine_cell_straddling_two_coarse_cells_takes_their_mean() -> None:
    # Coarse cells [0, 0.25] = 1 and [0.25, 0.5] = 3; fine cell [0.2, 0.3] is half each.
    values = np.tile([1.0, 3.0, 5.0, 7.0], (4, 1))
    out = conservative_regrid(field(values, 44 + COARSE, COARSE), 44 + FINE, FINE)
    np.testing.assert_allclose(out.isel(latitude=0).values[:3], [1.0, 1.0, 2.0])


def test_target_cells_outside_the_source_are_refused() -> None:
    src = field(np.ones((4, 4)), 44 + COARSE, COARSE)
    with pytest.raises(ValueError, match="longitude"):
        conservative_regrid(src, 44 + FINE, FINE + 0.1)


def test_other_dimensions_are_kept() -> None:
    values = np.arange(2 * 3 * 4 * 4, dtype="float64").reshape(2, 3, 4, 4)
    src = xr.DataArray(
        values,
        dims=("member", "step", "latitude", "longitude"),
        coords={"member": [0, 1], "step": [6, 12, 18], "latitude": 44 + COARSE},
        attrs={"units": "mm"},
    ).assign_coords(longitude=COARSE)
    out = conservative_regrid(src, 44 + FINE, FINE)
    assert out.dims == ("member", "step", "latitude", "longitude")
    assert out.shape == (2, 3, 10, 10)
    assert out.attrs == {"units": "mm"}
    np.testing.assert_allclose(
        out.isel(member=1, step=2).values[0, 0], values[1, 2, 0, 0]
    )
