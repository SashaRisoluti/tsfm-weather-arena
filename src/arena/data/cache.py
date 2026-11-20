"""Local Zarr cache of regional subsets.

Layout under ``<cache_dir>/<region>/``::

    pardig/archive.zarr             2016-2025, cut from the time-series store
    pardig/frames/YYYYMMDD.zarr     one day of 15-minute frames
    satweave/YYYYMMDD.zarr          one day of hourly frames
    aifs_ens/YYYYMMDDHH.zarr        one forecast run
    static/elevation.zarr           cell-mean terrain elevation

Every store is written to a temporary folder and then renamed, so a store
that exists is complete. Precipitation is stored in mm/h.
"""

import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from omegaconf import DictConfig

from arena.data.region import Region, select_region
from arena.data.sources import elevation, pardig, satweave
from arena.paths import project_root

log = logging.getLogger(__name__)

_FRAME_SOURCES = {"pardig": pardig, "satweave": satweave}


def cache_root(cfg: DictConfig) -> Path:
    base = Path(cfg.data.cache_dir)
    if not base.is_absolute():
        base = project_root() / base
    return base / str(cfg.region.name)


def write_atomic(ds: xr.Dataset, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ds.to_zarr(tmp, mode="w", zarr_format=3, consolidated=False)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    tmp.replace(path)


def _concat(parts: list[xr.Dataset], dim: str = "time") -> xr.Dataset:
    if len(parts) == 1:
        return parts[0]
    return xr.concat(
        parts,
        dim=dim,
        data_vars="all",
        coords="minimal",
        compat="override",
        join="exact",
        combine_attrs="drop_conflicts",
    )


def cache_pardig_archive(cfg: DictConfig) -> Path:
    """Cut the region out of the multi-year archive, once."""
    path = cache_root(cfg) / "pardig" / "archive.zarr"
    if path.exists():
        return path
    source = cfg.data.pardig
    region = Region.from_config(cfg.region)
    with pardig.open_archive(cfg) as archive:
        sub = select_region(archive, region, cfg.data.grid_step)
        sub[source.variable] = pardig.to_mm_per_hour(
            sub[source.variable], source.rate_to_mm_per_hour
        )
        # One whole series per 0.5 degree tile, as in the source, wherever
        # the box starts relative to the source tiles.
        sub = sub.drop_encoding().chunk({"time": -1, "latitude": 5, "longitude": 5})
        write_atomic(sub, path)
    log.info("pardig archive cached")
    return path


def cache_elevation(cfg: DictConfig) -> Path:
    """Fetch the terrain elevation of the region, once."""
    path = cache_root(cfg) / "static" / "elevation.zarr"
    if not path.exists():
        heights = elevation.fetch(cfg, Region.from_config(cfg.region))
        write_atomic(heights.to_dataset(name="elevation"), path)
        log.info("elevation cached")
    return path


def _day_times(day: date, step_minutes: int) -> list[datetime]:
    start = datetime(day.year, day.month, day.day)
    return [start + timedelta(minutes=m) for m in range(0, 24 * 60, step_minutes)]


def _load_frame(
    cfg: DictConfig, source: str, time: datetime, region: Region
) -> xr.Dataset | None:
    try:
        with _FRAME_SOURCES[source].open_frame(cfg, time) as frame:
            return select_region(frame, region, cfg.data.grid_step).load()
    except FileNotFoundError:
        return None


def _fetch_day(
    cfg: DictConfig, source: str, day: date
) -> tuple[xr.Dataset | None, int]:
    """Regional data for one day on its full time axis, and how many steps
    of it are unavailable."""
    settings = cfg.data[source]
    region = Region.from_config(cfg.region)
    times = _day_times(day, settings.step_minutes)
    first_frame = datetime.fromisoformat(settings.frames_start)
    parts: list[xr.Dataset]
    if times[0] >= first_frame:
        with ThreadPoolExecutor(max_workers=8) as pool:
            loaded = pool.map(lambda t: _load_frame(cfg, source, t, region), times)
            parts = [frame for frame in loaded if frame is not None]
    elif source == "satweave":
        with satweave.open_archive(cfg) as archive:
            stored = archive.time.to_index()
            wanted = [t for t in times if t in stored]
            steps = archive.sel(time=wanted)
            parts = [select_region(steps, region, cfg.data.grid_step).load()]
    else:
        raise ValueError(
            f"{source} before {first_frame:%Y-%m-%d} is in the archive, "
            "not in daily frames"
        )
    if not parts:
        return None, len(times)
    data = _concat(parts)
    # A step whose chunk was never uploaded reads back as NaN everywhere.
    empty = xr.concat(
        [v.isnull().all(("latitude", "longitude")) for v in data.data_vars.values()],
        dim="variable",
    ).all("variable")
    data = data.isel(time=~empty.values)
    if data.sizes["time"] == 0:
        return None, len(times)
    missing = len(times) - data.sizes["time"]
    return data.reindex(time=times), missing


def cache_days(
    cfg: DictConfig, source: str, start: date, end: date, allow_gaps: bool = False
) -> list[Path]:
    """Cache each day from ``start`` to ``end`` that is not cached yet.

    A day with unavailable steps is skipped unless ``allow_gaps`` is set, in
    which case those steps are NaN and counted in ``missing_frames``.
    """
    if source not in _FRAME_SOURCES:
        raise ValueError(
            f"unknown source {source!r}; expected one of {sorted(_FRAME_SOURCES)}"
        )
    folder = cache_root(cfg) / source
    if source == "pardig":
        folder = folder / "frames"
    written = []
    day = start
    while day <= end:
        path = folder / f"{day:%Y%m%d}.zarr"
        if not path.exists():
            data, missing = _fetch_day(cfg, source, day)
            if data is None or (missing and not allow_gaps):
                log.warning(
                    "%s %s: %d steps unavailable, not cached", source, day, missing
                )
            else:
                if source == "pardig":
                    name = cfg.data.pardig.variable
                    factor = cfg.data.pardig.rate_to_mm_per_hour
                    data[name] = pardig.to_mm_per_hour(data[name], factor)
                data.attrs = {"missing_frames": missing}
                write_atomic(data.drop_encoding(), path)
                written.append(path)
                log.info("%s %s cached", source, day)
        day += timedelta(days=1)
    return written


def _open_run(cfg: DictConfig, store: Path) -> xr.Dataset:
    """One cached forecast run, shaped for stacking along ``init_time``."""
    run = xr.open_zarr(store, consolidated=False, chunks={})
    template = run[next(iter(run.data_vars))]
    for param in cfg.data.aifs_ens.params:
        if param not in run:
            # Not produced by the model version of this run.
            run[param] = xr.full_like(template, np.nan)
    run["model_version"] = ((), run.attrs["model_version"])
    return run.expand_dims("init_time")


def open_cached(cfg: DictConfig, source: str) -> xr.Dataset:
    """Everything cached for ``source`` in the configured region, lazily."""
    root = cache_root(cfg) / source
    if source == "pardig":
        stores = sorted((root / "frames").glob("*.zarr"))
        if (root / "archive.zarr").exists():
            stores.insert(0, root / "archive.zarr")
    else:
        stores = sorted(root.glob("*.zarr"))
    if not stores:
        raise FileNotFoundError(f"nothing cached for {source} in {cfg.region.name}")
    if source == "aifs_ens":
        return _concat([_open_run(cfg, s) for s in stores], dim="init_time")
    return _concat([xr.open_zarr(s, consolidated=False, chunks={}) for s in stores])
