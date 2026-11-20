"""Regional archive of AIFS ensemble runs.

ECMWF open data keeps past runs on public mirrors (Google Cloud and AWS)
from the first operational run on 2025-07-01 06 UTC, with no stated
retention. ``backfill`` copies a range of runs into the local cache and
``update`` catches up from the newest cached run to the newest disseminated
one, so the archive grows whenever the command is run.
"""

import csv
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiohttp
import fsspec
from omegaconf import DictConfig

from arena.data import cache
from arena.data.region import Region
from arena.data.sources import aifs_ens

log = logging.getLogger(__name__)


def init_times(cfg: DictConfig, start: datetime, end: datetime) -> list[datetime]:
    """Run times between ``start`` and ``end`` (inclusive) at the configured hours."""
    settings = cfg.data.aifs_ens
    start = max(start, datetime.fromisoformat(settings.first_init))
    hours = sorted(int(hour) for hour in settings.init_hours)
    times = []
    day = datetime(start.year, start.month, start.day)
    while day <= end:
        times += [day + timedelta(hours=hour) for hour in hours]
        day += timedelta(days=1)
    return [time for time in times if start <= time <= end]


def latest_available_init(cfg: DictConfig, now: datetime) -> datetime:
    """Newest run that counts as disseminated at ``now`` (UTC)."""
    limit = now - timedelta(hours=int(cfg.data.aifs_ens.availability_lag_hours))
    return init_times(cfg, limit - timedelta(days=1), limit)[-1]


def _run_path(cfg: DictConfig, init: datetime) -> Path:
    return cache.cache_root(cfg) / "aifs_ens" / f"{init:%Y%m%d%H}.zarr"


def backfill(cfg: DictConfig, start: datetime, end: datetime) -> list[Path]:
    """Cache every run from ``start`` to ``end`` that is not cached yet.

    A run that is not on the mirrors, or is there in an unexpected form, is
    logged and skipped, so one bad run does not stop a long back-fill.
    """
    region = Region.from_config(cfg.region)
    written = []
    for init in init_times(cfg, start, end):
        path = _run_path(cfg, init)
        if path.exists():
            continue
        try:
            run = aifs_ens.fetch_run(cfg, init, region)
        except (aifs_ens.RunNotAvailable, ValueError) as error:
            log.warning("skipped: %s", error)
            continue
        cache.write_atomic(run, path)
        written.append(path)
        log.info("run %s cached", f"{init:%Y-%m-%d %Hz}")
    return written


def update(cfg: DictConfig, now: datetime | None = None) -> list[Path]:
    """Cache the runs disseminated since the newest cached one."""
    if now is None:
        now = datetime.now(UTC).replace(tzinfo=None)
    folder = cache.cache_root(cfg) / "aifs_ens"
    cached = sorted(folder.glob("*.zarr"))
    if cached:
        newest = datetime.strptime(cached[-1].stem, "%Y%m%d%H")
        start = newest + timedelta(hours=1)
    else:
        start = datetime.fromisoformat(cfg.data.aifs_ens.first_init)
    return backfill(cfg, start, latest_available_init(cfg, now))


def _probe(cfg: DictConfig, init: datetime) -> tuple[bool, bool]:
    """Whether the first control file of a run is indexed, and lists ``cp``."""
    settings = cfg.data.aifs_ens
    step = int(settings.steps[0])
    for base in settings.base_urls:
        fs, _ = fsspec.core.url_to_fs(base)
        try:
            text = fs.cat_file(aifs_ens.run_url(base, init, step, "cf", "index"))
        except (OSError, aiohttp.ClientError, TimeoutError):
            continue
        params = {entry["param"] for entry in aifs_ens.parse_index(text.decode())}
        return "tp" in params, "cp" in params
    return False, False


def coverage(cfg: DictConfig, start: datetime, end: datetime) -> Path:
    """Scan the mirrors and write which runs exist, as ``coverage.csv``.

    Reads one small index file per run. ``has_cp`` marks the second model
    version, the first one to produce convective precipitation.
    """
    times = init_times(cfg, start, end)
    with ThreadPoolExecutor(max_workers=8) as pool:
        found = list(pool.map(lambda init: _probe(cfg, init), times))
    path = cache.cache_root(cfg) / "aifs_ens" / "coverage.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["init_time", "available", "has_cp"])
        for init, (available, has_cp) in zip(times, found, strict=True):
            writer.writerow([f"{init:%Y-%m-%dT%H:%M}", available, has_cp])
    return path
