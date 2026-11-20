"""Terrain elevation on the regional grid.

Copernicus DEM GLO-90 heights, read through the Open-Meteo elevation API
and averaged over each grid cell.
"""

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import fsspec
import numpy as np
import xarray as xr
from omegaconf import DictConfig

from arena.data.region import Region


def _get_json(url: str) -> dict[str, Any]:
    answer: dict[str, Any] = json.loads(fsspec.filesystem("https").cat_file(url))
    return answer


def _axis(lo: float, hi: float, step: float) -> np.ndarray:
    return np.asarray(np.round(lo + step * np.arange(round((hi - lo) / step) + 1), 6))


def fetch(
    cfg: DictConfig,
    region: Region,
    get_json: Callable[[str], dict[str, Any]] | None = None,
) -> xr.DataArray:
    """Cell-mean elevation in metres for every grid point of ``region``.

    Each cell is sampled on a regular ``subsamples`` x ``subsamples`` lattice,
    because a single point of a 90 m model is a poor summary of a 10 km cell
    in the mountains.
    """
    get_json = get_json or _get_json
    settings = cfg.data.elevation
    step = float(cfg.data.grid_step)
    n = int(settings.subsamples)
    lat = _axis(region.lat_min, region.lat_max, step)
    lon = _axis(region.lon_min, region.lon_max, step)
    # Offsets of the sample points from the cell centre, in degrees.
    offsets = ((np.arange(n) + 0.5) / n - 0.5) * step
    sub_lat = (lat[:, None] + offsets[None, :]).ravel()
    sub_lon = (lon[:, None] + offsets[None, :]).ravel()
    points_lat, points_lon = (
        a.ravel() for a in np.meshgrid(sub_lat, sub_lon, indexing="ij")
    )

    def ask(start: int) -> list[float]:
        chunk = slice(start, start + int(settings.batch_size))
        url = (
            f"{settings.url}"
            f"?latitude={','.join(f'{v:.4f}' for v in points_lat[chunk])}"
            f"&longitude={','.join(f'{v:.4f}' for v in points_lon[chunk])}"
        )
        heights = get_json(url)["elevation"]
        if len(heights) != points_lat[chunk].size:
            raise ValueError("elevation service returned a different number of points")
        return [np.nan if h is None else float(h) for h in heights]

    with ThreadPoolExecutor(max_workers=4) as pool:
        batches = pool.map(ask, range(0, points_lat.size, int(settings.batch_size)))
        heights = np.concatenate([np.asarray(batch) for batch in batches])
    cells = heights.reshape(lat.size, n, lon.size, n).mean(axis=(1, 3))
    return xr.DataArray(
        cells,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
        attrs={
            "units": "m",
            "long_name": "Cell-mean terrain elevation",
            "source": "Copernicus DEM GLO-90 via the Open-Meteo elevation API",
        },
    )
