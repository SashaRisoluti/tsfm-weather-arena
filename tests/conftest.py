"""Synthetic stand-ins for the remote stores, shaped like the real ones."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from omegaconf import DictConfig, OmegaConf

from arena.config import load_config

from . import grib_mirror

# A patch of the 0.1 degree grid that contains the Alps box with a margin.
LAT = np.round(40.0 + 0.1 * np.arange(101), 6)
LON = np.round(0.1 * np.arange(201), 6)
STAMP = "{time:%Y%m%dT%H%M%S}"

# 2026-01-02 frames that are deliberately unusable.
ABSENT = datetime(2026, 1, 2, 12, 0)
ALL_NAN = datetime(2026, 1, 2, 13, 0)
NO_MARKER = datetime(2026, 1, 2, 14, 0)


def pardig_rate(step_of_day: int) -> np.ndarray:
    """Rate in m/h: 0.1 mm/h per step of the day plus 0.001 mm/h per row."""
    rows = np.arange(LAT.size, dtype="float32")[:, None]
    field = (step_of_day + 1) * 1e-4 + rows * 1e-6
    return np.broadcast_to(field, (LAT.size, LON.size)).astype("float32")


def _write(ds: xr.Dataset, path: Path) -> None:
    ds.to_zarr(path, mode="w", zarr_format=3)


def _pardig(root: Path) -> None:
    times = [datetime(2025, 12, 31) + timedelta(minutes=15 * k) for k in range(96)]
    archive = xr.Dataset(
        {
            "precipitation": (
                ("time", "latitude", "longitude"),
                np.stack([pardig_rate(k) for k in range(96)]),
                {"units": "m h**-1", "step_type": "instant"},
            )
        },
        coords={"time": times, "latitude": LAT, "longitude": LON},
    )
    _write(
        archive.chunk({"time": 96, "latitude": 5, "longitude": 5}),
        root / "archive.zarr",
    )
    for day in (datetime(2026, 1, 1), datetime(2026, 1, 2)):
        for k in range(96):
            time = day + timedelta(minutes=15 * k)
            if time == ABSENT:
                continue
            values = pardig_rate(k)
            if time == ALL_NAN:
                values = np.full_like(values, np.nan)
            frame = xr.Dataset(
                {"precipitation": (("latitude", "longitude"), values)},
                coords={"latitude": LAT, "longitude": LON, "time": time},
            )
            folder = root / "frames" / STAMP.format(time=time)
            _write(frame, folder / "predictions.zarr")
            if time != NO_MARKER:
                (folder / "success").touch()


def _satweave(root: Path) -> None:
    # Stored north to south, with the float noise found in the real coordinate.
    lat = LAT[::-1] + 1e-14
    shape = (lat.size, LON.size)

    def bands(times: list[datetime]) -> xr.Dataset:
        hours = np.array([t.hour + t.minute / 60 for t in times], dtype="float32")
        cube = np.broadcast_to(hours[:, None, None], (len(times), *shape))
        dims = ("time", "latitude", "longitude")
        return xr.Dataset(
            {
                "mosaic_11200_nm": (dims, 250.0 + cube, {"units": "K"}),
                "mosaic_6200_nm": (dims, 220.0 + cube, {"units": "K"}),
                "mosaic_10400_nm": (dims, np.full(cube.shape, np.nan, "float32")),
            },
            coords={"time": times, "latitude": lat, "longitude": LON},
        )

    half_hours = [datetime(2025, 12, 31) + timedelta(minutes=30 * k) for k in range(48)]
    _write(bands(half_hours).chunk({"time": 1}), root / "archive.zarr")
    for hour in range(24):
        time = datetime(2026, 1, 1, hour)
        folder = root / "frames" / STAMP.format(time=time)
        _write(bands([time]), folder / "mosaic.zarr")
        (folder / "success").touch()


@pytest.fixture(scope="session")
def remote(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("remote")
    _pardig(root / "pardig")
    _satweave(root / "satweave")
    return root


@pytest.fixture(scope="session")
def mirrors(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return grib_mirror.build_mirrors(tmp_path_factory.mktemp("mirrors"))


@pytest.fixture
def synthetic(remote: Path, mirrors: tuple[Path, Path], tmp_path: Path) -> DictConfig:
    """Project config pointed at the synthetic stores and a fresh cache."""
    cfg = load_config()
    base = remote.as_posix()
    updates = {
        "data.aifs_ens.base_urls": [m.as_posix() for m in mirrors],
        "data.aifs_ens.members": grib_mirror.MEMBERS,
        "data.aifs_ens.steps": list(grib_mirror.STEPS),
        "data.cache_dir": (tmp_path / "cache").as_posix(),
        "data.pardig.archive_url": f"{base}/pardig/archive.zarr",
        "data.pardig.frame_url": f"{base}/pardig/frames/{STAMP}/predictions.zarr",
        "data.satweave.archive_url": f"{base}/satweave/archive.zarr",
        "data.satweave.frame_url": f"{base}/satweave/frames/{STAMP}/mosaic.zarr",
    }
    for key, value in updates.items():
        OmegaConf.update(cfg, key, value)
    return cfg
