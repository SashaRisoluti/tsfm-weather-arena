"""AIFS ensemble forecasts from ECMWF open data.

Each run has one GRIB file per step and member group (``cf`` control, ``pf``
perturbed members), with an index of byte ranges, on a 0.25 degree global
grid. Precipitation is accumulated from the start of the run in kg m-2,
which is mm. Only the requested messages are downloaded.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import aiohttp
import eccodes
import fsspec
import numpy as np
import xarray as xr
from omegaconf import DictConfig

from arena.data.region import Region, select_region

log = logging.getLogger(__name__)

_DIMS = ("member", "step", "latitude", "longitude")
_GRIDS: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}
_GRID_KEYS = (
    "Ni",
    "Nj",
    "latitudeOfFirstGridPointInDegrees",
    "latitudeOfLastGridPointInDegrees",
    "longitudeOfFirstGridPointInDegrees",
    "longitudeOfLastGridPointInDegrees",
)


class RunNotAvailable(Exception):
    """The run could not be read from any configured mirror."""


def run_url(base: str, init: datetime, step: int, kind: str, ext: str) -> str:
    return (
        f"{base}/{init:%Y%m%d}/{init:%H}z/aifs-ens/0p25/enfo/"
        f"{init:%Y%m%d%H}0000-{step}h-enfo-{kind}.{ext}"
    )


def parse_index(text: str) -> list[dict[str, Any]]:
    """One JSON object per line; ``_offset`` and ``_length`` locate a message."""
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def decode(message: bytes) -> xr.DataArray:
    """Decode one GRIB message on the grid as stored."""
    handle = eccodes.codes_new_from_message(message)
    try:
        grid = tuple(eccodes.codes_get(handle, key) for key in _GRID_KEYS)
        shape = (int(grid[1]), int(grid[0]))
        if grid not in _GRIDS:
            # Same geometry for every message of a run, so worked out once.
            lat = eccodes.codes_get_array(handle, "latitudes").reshape(shape)[:, 0]
            lon = eccodes.codes_get_array(handle, "longitudes").reshape(shape)[0, :]
            _GRIDS[grid] = (lat.copy(), lon.copy())
        lat, lon = _GRIDS[grid]
        values = eccodes.codes_get_values(handle).reshape(shape)
        attrs = {
            "param": str(eccodes.codes_get(handle, "shortName")),
            "step": int(eccodes.codes_get(handle, "endStep")),
            "model_version": int(
                eccodes.codes_get(handle, "generatingProcessIdentifier")
            ),
            "units": str(eccodes.codes_get(handle, "units")),
        }
    finally:
        eccodes.codes_release(handle)
    return xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
        attrs=attrs,
    )


def _download(
    fs: Any, base: str, settings: Any, init: datetime, step: int, kind: str
) -> list[tuple[str, int, bytes]]:
    """Requested messages of one file as (parameter, member, bytes)."""
    index = parse_index(fs.cat_file(run_url(base, init, step, kind, "index")).decode())
    wanted = [entry for entry in index if entry["param"] in settings.params]
    if not wanted:
        return []
    grib = run_url(base, init, step, kind, "grib2")
    starts = [entry["_offset"] for entry in wanted]
    ends = [entry["_offset"] + entry["_length"] for entry in wanted]
    blobs = fs.cat_ranges([grib] * len(wanted), starts, ends, on_error="raise")
    return [
        (entry["param"], int(entry.get("number", 0)), blob)
        for entry, blob in zip(wanted, blobs, strict=True)
    ]


def _fetch_from(base: str, settings: Any, init: datetime, region: Region) -> xr.Dataset:
    label = f"run {init:%Y-%m-%d %Hz}"
    fs, _ = fsspec.core.url_to_fs(base)
    steps = [int(step) for step in settings.steps]
    members = list(range(int(settings.members)))
    jobs = [(step, kind) for step in steps for kind in ("cf", "pf")]
    with ThreadPoolExecutor(max_workers=4) as pool:
        downloads = list(
            pool.map(lambda job: _download(fs, base, settings, init, *job), jobs)
        )

    # Decoded on the calling thread: ecCodes is not assumed to be thread-safe.
    fields: dict[str, dict[tuple[int, int], xr.DataArray]] = {}
    for (step, _), messages in zip(jobs, downloads, strict=True):
        for param, member, blob in messages:
            field = decode(blob)
            if field.attrs["step"] != step or field.attrs["units"] != "kg m**-2":
                raise ValueError(f"{label}: unexpected {param} message at step {step}")
            cut = select_region(field, region, settings.grid_step)
            fields.setdefault(param, {})[(member, step)] = cut

    data_vars = {}
    versions = set()
    for param in settings.params:
        # A parameter the model version does not produce is simply absent.
        got = fields.get(param)
        if not got:
            continue
        absent = [(m, s) for m in members for s in steps if (m, s) not in got]
        if absent:
            raise ValueError(
                f"{label}: {param} lacks {len(absent)} of "
                f"{len(members) * len(steps)} member-step fields"
            )
        cube = np.stack([np.stack([got[m, s].values for s in steps]) for m in members])
        long_name = f"{param} accumulated from the start of the run"
        attrs = {"units": "mm", "long_name": long_name}
        data_vars[param] = (_DIMS, cube.astype("float32"), attrs)
        versions |= {field.attrs["model_version"] for field in got.values()}
    if not data_vars:
        raise ValueError(f"{label}: none of {list(settings.params)} in the index")
    if len(versions) != 1:
        raise ValueError(f"{label}: mixed model versions {sorted(versions)}")
    grid = next(iter(next(iter(fields.values())).values()))
    return xr.Dataset(
        data_vars,
        coords={
            "member": members,
            "step": steps,
            "latitude": grid.latitude,
            "longitude": grid.longitude,
            "init_time": np.datetime64(init, "ns"),
        },
        attrs={"model_version": versions.pop()},
    )


def fetch_run(cfg: DictConfig, init: datetime, region: Region) -> xr.Dataset:
    """Regional accumulations of one run, all members and configured steps.

    Mirrors are tried in order. Raises ``RunNotAvailable`` if none has the
    complete run, and ``ValueError`` if a mirror has it in an unexpected form.
    """
    settings = cfg.data.aifs_ens
    for number, base in enumerate(settings.base_urls, start=1):
        try:
            return _fetch_from(base, settings, init, region)
        except (OSError, aiohttp.ClientError, TimeoutError) as error:
            log.info(
                "run %s not read from mirror %d (%s)",
                f"{init:%Y-%m-%d %Hz}",
                number,
                type(error).__name__,
            )
    raise RunNotAvailable(f"run {init:%Y-%m-%d %Hz} could not be read from any mirror")


def deaccumulate(ds: xr.Dataset) -> xr.Dataset:
    """Totals per step from accumulations since the start of the run.

    The first step must be one interval after the start and the steps
    contiguous. Small negative differences, which come from packing each
    message separately, are clipped to zero.
    """
    steps = ds.step.values
    if np.any(np.diff(steps) != steps[0]):
        raise ValueError("steps must be contiguous from one interval after the start")
    totals = xr.concat(
        [ds.isel(step=[0]), ds.diff("step")],
        dim="step",
        data_vars="all",
        coords="minimal",
        compat="override",
        join="exact",
    )
    return totals.clip(min=0.0, keep_attrs=True)
