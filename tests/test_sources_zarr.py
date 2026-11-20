from datetime import datetime

import numpy as np
import pytest
import xarray as xr
from omegaconf import DictConfig

from arena.data.sources import pardig, satweave

from .conftest import ABSENT, NO_MARKER


def test_pardig_archive_opens_lazily(synthetic: DictConfig) -> None:
    with pardig.open_archive(synthetic) as ds:
        assert list(ds.data_vars) == ["precipitation"]
        assert ds.precipitation.dims == ("time", "latitude", "longitude")
        assert ds.precipitation.chunks is not None


def test_pardig_frame_has_a_time_dimension(synthetic: DictConfig) -> None:
    time = datetime(2026, 1, 1, 0, 15)
    with pardig.open_frame(synthetic, time) as ds:
        assert dict(ds.sizes) == {"time": 1, "latitude": 101, "longitude": 201}
        assert ds.time.values[0] == np.datetime64(time)


@pytest.mark.parametrize("time", [ABSENT, NO_MARKER])
def test_unpublished_frame_raises_without_naming_a_path(
    synthetic: DictConfig, time: datetime
) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        pardig.open_frame(synthetic, time)
    assert "2026-01-02" in str(excinfo.value)
    assert "remote" not in str(excinfo.value)


def test_to_mm_per_hour_converts_and_labels() -> None:
    rate = xr.DataArray(np.array([0.0, 0.001, 0.075], "float32"), dims="time")
    out = pardig.to_mm_per_hour(rate, 1000.0)
    np.testing.assert_allclose(out.values, [0.0, 1.0, 75.0], rtol=1e-6)
    assert out.attrs["units"] == "mm h-1"
    assert out.dtype == np.float32


def test_satweave_keeps_only_the_configured_bands(synthetic: DictConfig) -> None:
    expected = ["mosaic_11200_nm", "mosaic_6200_nm"]
    with satweave.open_archive(synthetic) as archive:
        assert list(archive.data_vars) == expected
        assert archive.sizes["time"] == 48
    with satweave.open_frame(synthetic, datetime(2026, 1, 1, 5)) as frame:
        assert list(frame.data_vars) == expected
        assert frame.sizes["time"] == 1
