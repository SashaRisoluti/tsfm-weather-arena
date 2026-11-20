"""Terrain elevation on the regional grid.

Copernicus DEM GLO-90 heights, read through the Open-Meteo elevation API.
"""

import json
import logging
import time
from collections.abc import Callable
from typing import Any

import aiohttp
import fsspec
import numpy as np
import xarray as xr
from omegaconf import DictConfig

from arena.data.region import Region

log = logging.getLogger(__name__)


class RateLimited(Exception):
    """The service asked to wait ``wait`` seconds before the next request."""

    def __init__(self, wait: float) -> None:
        super().__init__(f"rate limited for {wait:.0f} s")
        self.wait = wait


def _get_json(url: str) -> dict[str, Any]:
    try:
        raw = fsspec.filesystem("https").cat_file(url)
    except aiohttp.ClientResponseError as error:
        if error.status == 429:
            wait = error.headers.get("Retry-After", "60") if error.headers else "60"
            raise RateLimited(float(wait)) from None
        raise
    answer: dict[str, Any] = json.loads(raw)
    return answer


def _axis(lo: float, hi: float, step: float) -> np.ndarray:
    return np.asarray(np.round(lo + step * np.arange(round((hi - lo) / step) + 1), 6))


def fetch(
    cfg: DictConfig,
    region: Region,
    get_json: Callable[[str], dict[str, Any]] | None = None,
) -> xr.DataArray:
    """Elevation in metres for every grid point of ``region``.

    Each cell is the mean of a ``subsamples`` x ``subsamples`` lattice of
    points. Requests are sent one at a time and wait when the service says
    its request limit is reached.
    """
    get_json = get_json or _get_json
    settings = cfg.data.elevation
    step = float(cfg.data.grid_step)
    n = int(settings.subsamples)
    size = int(settings.batch_size)
    lat = _axis(region.lat_min, region.lat_max, step)
    lon = _axis(region.lon_min, region.lon_max, step)
    # Offsets of the sample points from the cell centre, in degrees.
    offsets = ((np.arange(n) + 0.5) / n - 0.5) * step
    sub_lat = (lat[:, None] + offsets[None, :]).ravel()
    sub_lon = (lon[:, None] + offsets[None, :]).ravel()
    points_lat, points_lon = (
        a.ravel() for a in np.meshgrid(sub_lat, sub_lon, indexing="ij")
    )

    heights: list[float] = []
    for start in range(0, points_lat.size, size):
        chunk = slice(start, start + size)
        url = (
            f"{settings.url}"
            f"?latitude={','.join(f'{v:.4f}' for v in points_lat[chunk])}"
            f"&longitude={','.join(f'{v:.4f}' for v in points_lon[chunk])}"
        )
        while True:
            try:
                answer = get_json(url)["elevation"]
                break
            except RateLimited as limit:
                if limit.wait > float(settings.max_wait_seconds):
                    raise RuntimeError(
                        "elevation service request limit reached; "
                        f"retry in about {limit.wait / 60:.0f} minutes"
                    ) from None
                log.info("elevation service busy, waiting %.0f s", limit.wait + 1)
                time.sleep(limit.wait + 1)
        if len(answer) != points_lat[chunk].size:
            raise ValueError("elevation service returned a different number of points")
        heights += [np.nan if h is None else float(h) for h in answer]

    cells = np.asarray(heights).reshape(lat.size, n, lon.size, n).mean(axis=(1, 3))
    return xr.DataArray(
        cells,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
        attrs={
            "units": "m",
            "long_name": "Terrain elevation",
            "source": "Copernicus DEM GLO-90 via the Open-Meteo elevation API",
            "samples_per_cell": n * n,
        },
    )
