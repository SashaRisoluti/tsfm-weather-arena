"""Common grid and conservative regridding between regular lat-lon grids.

Conservative regridding gives every target cell the area-weighted mean of the
source cells it overlaps, so the total of an amount per unit area (such as
precipitation in mm) over a shared domain is unchanged. On regular grids the
weights separate into a longitude factor (overlap in degrees) and a latitude
factor (overlap in sin(latitude), proportional to area on the sphere).
"""

from collections.abc import Callable

import numpy as np
import xarray as xr

from arena.data.region import Region


def common_grid(region: Region, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Ascending latitude and longitude nodes of ``region``, bounds inclusive."""

    def nodes(lo: float, hi: float) -> np.ndarray:
        count = int(round((hi - lo) / step)) + 1
        return np.round(lo + step * np.arange(count), 6)

    return (
        nodes(region.lat_min, region.lat_max),
        nodes(region.lon_min, region.lon_max),
    )


def _edges(centres: np.ndarray, name: str) -> np.ndarray:
    steps = np.diff(centres)
    if centres.size < 2 or not np.allclose(steps, steps[0]) or steps[0] <= 0:
        raise ValueError(f"{name} must be regular and ascending")
    half = steps[0] / 2
    return np.concatenate([centres - half, [centres[-1] + half]])


def _weights(
    src: np.ndarray,
    dst: np.ndarray,
    measure: Callable[[np.ndarray], np.ndarray],
    name: str,
) -> np.ndarray:
    """Matrix (target, source) of overlap fractions of each target cell."""
    se, de = _edges(src, name), _edges(dst, name)
    lo = np.maximum(de[:-1, None], se[None, :-1])
    hi = np.minimum(de[1:, None], se[None, 1:])
    overlap = np.where(hi > lo, measure(hi) - measure(lo), 0.0)
    size = measure(de[1:]) - measure(de[:-1])
    if not np.allclose(overlap.sum(axis=1), size, rtol=1e-9, atol=1e-12):
        raise ValueError(f"target {name} cells reach outside the source grid")
    return overlap / size[:, None]


def _sin_lat(lat: np.ndarray) -> np.ndarray:
    return np.sin(np.radians(lat))


def cell_areas(da: xr.DataArray) -> xr.DataArray:
    """Cell areas of a regular grid, in units of the Earth's radius squared."""
    lat, lon = (
        _edges(da.latitude.values, "latitude"),
        _edges(da.longitude.values, "longitude"),
    )
    band = np.diff(_sin_lat(lat))
    width = np.radians(np.diff(lon))
    return xr.DataArray(
        np.outer(band, width),
        dims=("latitude", "longitude"),
        coords={"latitude": da.latitude, "longitude": da.longitude},
    )


def conservative_regrid(
    da: xr.DataArray, latitude: np.ndarray, longitude: np.ndarray
) -> xr.DataArray:
    """Remap ``da`` to the cells centred on ``latitude`` x ``longitude``.

    Both grids must be regular and ascending, and every target cell must lie
    inside the source grid. Fields must be complete: a missing value spreads
    to every target cell in its source row and column.
    """
    w_lat = xr.DataArray(
        _weights(da.latitude.values, np.asarray(latitude), _sin_lat, "latitude"),
        dims=("lat_out", "latitude"),
    )
    w_lon = xr.DataArray(
        _weights(da.longitude.values, np.asarray(longitude), lambda x: x, "longitude"),
        dims=("lon_out", "longitude"),
    )
    order = da.dims
    out = xr.dot(xr.dot(da, w_lat, dim="latitude"), w_lon, dim="longitude")
    out = out.rename(lat_out="latitude", lon_out="longitude").assign_coords(
        latitude=np.asarray(latitude), longitude=np.asarray(longitude)
    )
    out.attrs = da.attrs
    return out.transpose(*order)
