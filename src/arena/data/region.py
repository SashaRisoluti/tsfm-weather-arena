"""Regions and box selection on regular latitude-longitude grids."""

from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import xarray as xr

T = TypeVar("T", xr.Dataset, xr.DataArray)


@dataclass(frozen=True)
class Region:
    """Latitude-longitude box, bounds inclusive, longitudes in [-180, 180)."""

    name: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    @classmethod
    def from_config(cls, cfg: Any) -> "Region":
        return cls(
            name=str(cfg.name),
            lon_min=float(cfg.lon_min),
            lon_max=float(cfg.lon_max),
            lat_min=float(cfg.lat_min),
            lat_max=float(cfg.lat_max),
        )


def _snap(values: Any, step: float) -> np.ndarray:
    """Nearest grid node, which removes float noise from stored coordinates."""
    nodes = np.round(np.asarray(values, dtype="float64") / step) * step
    return np.asarray(np.round(nodes, 6))


def _box(coord: np.ndarray, lo: float, hi: float, step: float, name: str) -> np.ndarray:
    """Positions of the nodes inside [lo, hi], by increasing coordinate."""
    tol = step * 1e-3
    inside = np.flatnonzero((coord >= lo - tol) & (coord <= hi + tol))
    inside = inside[np.argsort(coord[inside], kind="stable")]
    found = coord[inside]
    covered = (
        inside.size > 0
        and found[0] < lo + step - tol
        and found[-1] > hi - step + tol
        and bool(np.allclose(np.diff(found), step))
    )
    if not covered:
        raise ValueError(f"{name} range {lo}..{hi} is not fully inside the grid")
    return inside


def _indexer(positions: np.ndarray) -> slice | np.ndarray:
    """A slice when the positions are contiguous, which keeps dask chunks whole."""
    steps = np.diff(positions)
    first, last = int(positions[0]), int(positions[-1])
    if np.all(steps == 1):
        return slice(first, last + 1)
    if np.all(steps == -1):
        return slice(first, last - 1 if last > 0 else None, -1)
    return positions


def select_region(obj: T, region: Region, step: float) -> T:
    """Cut ``region`` out of a regular grid with spacing ``step`` degrees.

    Accepts either latitude order and either longitude convention. The result
    has ascending latitude, ascending longitude in [-180, 180), and coordinates
    snapped to the grid.
    """
    lat = _snap(obj["latitude"].values, step)
    lon = _snap(obj["longitude"].values, step)
    lon = np.asarray(np.round((lon + 180.0) % 360.0 - 180.0, 6))
    lat_pos = _box(lat, region.lat_min, region.lat_max, step, "latitude")
    lon_pos = _box(lon, region.lon_min, region.lon_max, step, "longitude")
    out = obj.isel(latitude=_indexer(lat_pos), longitude=_indexer(lon_pos))
    return out.assign_coords(
        latitude=("latitude", lat[lat_pos], obj["latitude"].attrs),
        longitude=("longitude", lon[lon_pos], obj["longitude"].attrs),
    )
