"""PARDIG precipitation estimates (near-real-time product).

Values are instantaneous rates in m/h on a 0.1 degree global grid, one frame
every 15 minutes: a time-series-chunked archive up to 2025 and one store per
frame afterwards.
"""

from datetime import datetime

import xarray as xr
from omegaconf import DictConfig

from arena.data.sources import _store


def open_archive(cfg: DictConfig) -> xr.Dataset:
    source = cfg.data.pardig
    return _store.open_archive(source.archive_url)[[source.variable]]


def open_frame(cfg: DictConfig, time: datetime) -> xr.Dataset:
    source = cfg.data.pardig
    frame = _store.open_frame(source.frame_url, source.marker, time)
    return frame[[source.variable]]


def to_mm_per_hour(rate: xr.DataArray, factor: float) -> xr.DataArray:
    """Convert the source rate to mm/h; ``factor`` is 1000 for m/h."""
    out = rate * factor
    out.attrs = {"long_name": "Precipitation rate", "units": "mm h-1"}
    return out
