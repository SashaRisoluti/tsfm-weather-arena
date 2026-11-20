"""SatWeave geostationary mosaic.

Brightness temperatures in K and reflectances in % on a 0.1 degree global
grid stored north to south: a half-hourly archive up to 2025 and one store
per hour afterwards. Only the configured bands are read.
"""

from datetime import datetime

import xarray as xr
from omegaconf import DictConfig

from arena.data.sources import _store


def open_archive(cfg: DictConfig) -> xr.Dataset:
    source = cfg.data.satweave
    return _store.open_archive(source.archive_url)[list(source.bands)]


def open_frame(cfg: DictConfig, time: datetime) -> xr.Dataset:
    source = cfg.data.satweave
    frame = _store.open_frame(source.frame_url, source.marker, time)
    return frame[list(source.bands)]
