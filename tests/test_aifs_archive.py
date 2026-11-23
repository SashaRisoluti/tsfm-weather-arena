import csv
from datetime import datetime
from typing import Any

import numpy as np
import pytest
from omegaconf import DictConfig

from arena.data import aifs_archive, cache
from arena.data.sources import aifs_ens

from . import grib_mirror


def test_init_times_follow_the_configured_hours_and_first_run(
    synthetic: DictConfig,
) -> None:
    got = aifs_archive.init_times(
        synthetic, datetime(2025, 6, 30, 0), datetime(2025, 7, 2, 12)
    )
    assert got == [
        datetime(2025, 7, 1, 12),
        datetime(2025, 7, 2, 0),
        datetime(2025, 7, 2, 12),
    ]


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (datetime(2026, 6, 2, 12, 30), datetime(2026, 6, 2, 0)),
        (datetime(2026, 6, 2, 12, 0), datetime(2026, 6, 2, 0)),
        (datetime(2026, 6, 2, 11, 59), datetime(2026, 6, 1, 12)),
    ],
)
def test_latest_available_init_respects_the_dissemination_lag(
    synthetic: DictConfig, now: datetime, expected: datetime
) -> None:
    assert aifs_archive.latest_available_init(synthetic, now) == expected


def record_fetches(monkeypatch: pytest.MonkeyPatch) -> list[datetime]:
    asked: list[datetime] = []
    real = aifs_ens.fetch_run

    def spy(cfg: DictConfig, init: datetime, region: Any) -> Any:
        asked.append(init)
        return real(cfg, init, region)

    monkeypatch.setattr(aifs_ens, "fetch_run", spy)
    return asked


def test_backfill_skips_cached_runs_and_survives_bad_ones(
    synthetic: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = record_fetches(monkeypatch)
    start, end = grib_mirror.COMPLETE, grib_mirror.NOWHERE
    written = aifs_archive.backfill(synthetic, start, end)
    # the incomplete run and the two absent ones are skipped, not fatal
    assert [p.name for p in written] == ["2026060100.zarr", "2026060112.zarr"]
    assert len(asked) == 5

    asked.clear()
    assert aifs_archive.backfill(synthetic, start, end) == []
    assert grib_mirror.COMPLETE not in asked and grib_mirror.FALLBACK not in asked


def test_update_fetches_only_runs_that_are_not_cached(
    synthetic: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    aifs_archive.backfill(synthetic, grib_mirror.COMPLETE, grib_mirror.COMPLETE)
    asked = record_fetches(monkeypatch)
    written = aifs_archive.update(synthetic, now=datetime(2026, 6, 2, 1, 0))
    assert [p.name for p in written] == ["2026060112.zarr"]
    assert asked == [grib_mirror.FALLBACK]


def test_update_retries_a_run_skipped_earlier(synthetic: DictConfig) -> None:
    # a run between two cached ones was skipped, for example by a failed download
    for init in (grib_mirror.VERSION_1, grib_mirror.FALLBACK):
        aifs_archive.backfill(synthetic, init, init)
    written = aifs_archive.update(synthetic, now=datetime(2026, 6, 2, 1, 0))
    assert [p.name for p in written] == ["2026060100.zarr"]


def test_coverage_lists_every_init_time(synthetic: DictConfig) -> None:
    path = aifs_archive.coverage(synthetic, grib_mirror.VERSION_1, grib_mirror.COMPLETE)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert path == cache.cache_root(synthetic) / "aifs_ens" / "coverage.csv"
    assert len(rows) == 42  # 2026-05-11 12 UTC to 2026-06-01 00 UTC, twice a day
    assert rows[0] == {
        "init_time": "2026-05-11T12:00",
        "available": "True",
        "has_cp": "False",
    }
    assert rows[1]["available"] == "False"
    assert rows[-1] == {
        "init_time": "2026-06-01T00:00",
        "available": "True",
        "has_cp": "True",
    }


def test_open_cached_stacks_runs_and_keeps_the_model_version(
    synthetic: DictConfig,
) -> None:
    for init in (grib_mirror.COMPLETE, grib_mirror.VERSION_1):
        aifs_archive.backfill(synthetic, init, init)
    with cache.open_cached(synthetic, "aifs_ens") as ds:
        assert dict(ds.sizes) == {
            "init_time": 2,
            "member": 3,
            "step": 2,
            "latitude": 17,
            "longitude": 33,
        }
        assert list(ds.init_time.values) == [
            np.datetime64(grib_mirror.VERSION_1),
            np.datetime64(grib_mirror.COMPLETE),
        ]
        assert list(ds.model_version.values) == [1, 2]
        assert bool(ds.cp.isel(init_time=0).isnull().all())
        assert bool(ds.cp.isel(init_time=1).notnull().all())
        assert bool(ds.tp.notnull().all())
        assert ds.tp.attrs["units"] == "mm" and ds.cp.attrs["units"] == "mm"
