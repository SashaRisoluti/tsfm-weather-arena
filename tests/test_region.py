import dask.array
import numpy as np
import pytest
import xarray as xr

from arena.config import load_config
from arena.data.region import Region, select_region

ALPS = Region("alps", lon_min=6.0, lon_max=14.0, lat_min=43.5, lat_max=47.5)


def grid(lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """Field whose value encodes the position it was taken from."""
    values = np.arange(lat.size * lon.size, dtype="float64").reshape(lat.size, lon.size)
    return xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
    )


def test_region_from_config() -> None:
    assert Region.from_config(load_config().region) == ALPS


def test_ascending_latitude_and_0_360_longitude() -> None:
    lat = np.arange(1801) * 0.1 - 90.0
    lon = np.arange(3600) * 0.1
    out = select_region(grid(lat, lon), ALPS, step=0.1)
    assert out.shape == (41, 81)
    np.testing.assert_array_equal(
        out.latitude.values, np.round(43.5 + 0.1 * np.arange(41), 6)
    )
    np.testing.assert_array_equal(
        out.longitude.values, np.round(6.0 + 0.1 * np.arange(81), 6)
    )
    # 43.5 N is row 1335, 6.0 E is column 60 of the source grid
    assert out.values[0, 0] == 1335 * 3600 + 60
    assert out.values[-1, -1] == 1375 * 3600 + 140


def test_descending_latitude_with_float_noise() -> None:
    lat = 90.0 - np.arange(1801) * 0.1 + 1e-14
    lon = np.arange(3600) * 0.1
    out = select_region(grid(lat, lon), ALPS, step=0.1)
    assert out.shape == (41, 81)
    assert out.latitude.values[0] == 43.5
    assert out.latitude.values[-1] == 47.5
    assert np.all(np.diff(out.latitude.values) > 0)
    # 43.5 N is row 465 when counting from the north pole
    assert out.values[0, 0] == 465 * 3600 + 60
    assert out.values[-1, 0] == 425 * 3600 + 60


def test_quarter_degree_grid_starting_at_180() -> None:
    lat = 90.0 - np.arange(721) * 0.25
    lon = (180.0 + np.arange(1440) * 0.25) % 360.0
    out = select_region(grid(lat, lon), ALPS, step=0.25)
    assert out.shape == (17, 33)
    np.testing.assert_array_equal(out.longitude.values, 6.0 + 0.25 * np.arange(33))
    np.testing.assert_array_equal(out.latitude.values, 43.5 + 0.25 * np.arange(17))
    # 6.0 E sits 186 degrees after the first column
    assert out.values[0, 0] == 186 * 1440 + 744


def test_box_across_the_prime_meridian() -> None:
    lat = np.arange(1801) * 0.1 - 90.0
    lon = np.arange(3600) * 0.1
    west = Region("west", lon_min=-0.2, lon_max=0.2, lat_min=45.0, lat_max=45.1)
    out = select_region(grid(lat, lon), west, step=0.1)
    np.testing.assert_allclose(out.longitude.values, [-0.2, -0.1, 0.0, 0.1, 0.2])
    np.testing.assert_array_equal(
        out.values[0], 1350 * 3600 + np.array([3598, 3599, 0, 1, 2])
    )


def test_box_not_inside_grid_raises() -> None:
    lat = 44.0 + np.arange(31) * 0.1
    lon = np.arange(200) * 0.1
    with pytest.raises(ValueError):
        select_region(grid(lat, lon), ALPS, step=0.1)


def test_lazy_input_stays_lazy_and_dataset_is_supported() -> None:
    lat = np.arange(1801) * 0.1 - 90.0
    lon = np.arange(3600) * 0.1
    lazy = xr.DataArray(
        dask.array.zeros((4, 1801, 3600), chunks=(4, 5, 5)),
        dims=("time", "latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
    )
    out = select_region(lazy.to_dataset(name="x"), ALPS, step=0.1)
    assert isinstance(out, xr.Dataset)
    assert out.x.shape == (4, 41, 81)
    assert out.x.chunks is not None
