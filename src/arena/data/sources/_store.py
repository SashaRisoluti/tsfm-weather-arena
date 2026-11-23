"""Access shared by the sources published as a Zarr archive plus one store
per timestamp."""

from datetime import datetime

import fsspec
import numpy as np
import xarray as xr
import zarr.errors


def open_archive(url: str) -> xr.Dataset:
    """Open a multi-year store lazily, keeping its native chunks."""
    return xr.open_zarr(url, chunks={}, consolidated=True)


def open_frame(url_template: str, marker: str, time: datetime) -> xr.Dataset:
    """Open the store of one timestamp, with a ``time`` dimension of length 1.

    Raises ``FileNotFoundError`` if the frame is absent or its upload has not
    been marked complete. A store without its chunk reads back as NaN rather
    than failing, so the marker is checked first. Any other read error is
    raised as it is: a frame that could not be read is not a missing frame.
    """
    url = url_template.format(time=time)
    fs, path = fsspec.core.url_to_fs(url)
    try:
        # Read, not tested for existence: that would report an error as absence.
        fs.cat_file(f"{path.rsplit('/', 1)[0]}/{marker}")
        ds = xr.open_zarr(url, chunks=None, consolidated=True)
    except (FileNotFoundError, zarr.errors.GroupNotFoundError):
        raise FileNotFoundError(
            f"no complete frame for {time:%Y-%m-%dT%H:%M}"
        ) from None
    if "time" not in ds.dims:
        ds = ds.expand_dims("time")
    if ds.time.values[0] != np.datetime64(time):
        raise ValueError(f"frame for {time:%Y-%m-%dT%H:%M} holds another time")
    return ds
