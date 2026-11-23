import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from omegaconf import DictConfig, OmegaConf

from arena.data.region import Region
from arena.data.sources import aifs_ens

from . import grib_mirror

ALPS = Region("alps", lon_min=6.0, lon_max=14.0, lat_min=43.5, lat_max=47.5)
# Rows of the synthetic patch, counted from 50 N in 0.25 degree steps.
ROW_43_5, ROW_47_5 = 26, 10


def test_run_url_follows_the_open_data_layout() -> None:
    url = aifs_ens.run_url(
        "https://host/bucket", datetime(2026, 6, 1, 0), 24, "pf", "grib2"
    )
    assert url == (
        "https://host/bucket/20260601/00z/aifs-ens/0p25/enfo/"
        "20260601000000-24h-enfo-pf.grib2"
    )


def test_parse_index_reads_one_entry_per_line() -> None:
    text = (
        '{"param": "tp", "number": "3", "_offset": 10, "_length": 5}\n'
        "\n"
        '{"param": "cp", "_offset": 15, "_length": 7}\n'
    )
    entries = aifs_ens.parse_index(text)
    assert [e["param"] for e in entries] == ["tp", "cp"]
    assert entries[0]["_offset"] + entries[0]["_length"] == entries[1]["_offset"]


def test_decode_returns_the_stored_grid_and_its_metadata() -> None:
    raw = grib_mirror.message("tp", 2, 12, grib_mirror.COMPLETE, version=2)
    out = aifs_ens.decode(raw)
    assert out.shape == (grib_mirror.NJ, grib_mirror.NI)
    assert (out.latitude.values[0], out.latitude.values[-1]) == (50.0, 40.0)
    assert (out.longitude.values[0], out.longitude.values[-1]) == (0.0, 20.0)
    assert out.attrs["param"] == "tp"
    assert out.attrs["member"] == 2
    assert out.attrs["step"] == 12
    assert out.attrs["model_version"] == 2
    assert out.attrs["units"] == "kg m**-2"
    np.testing.assert_allclose(out.values, grib_mirror.field("tp", 2, 12), atol=1e-3)


def test_fetch_run_assembles_members_and_steps(synthetic: DictConfig) -> None:
    ds = aifs_ens.fetch_run(synthetic, grib_mirror.COMPLETE, ALPS)
    assert dict(ds.sizes) == {"member": 3, "step": 2, "latitude": 17, "longitude": 33}
    assert list(ds.member.values) == [0, 1, 2]
    assert list(ds.step.values) == [6, 12]
    assert ds.latitude.values[0] == 43.5 and ds.longitude.values[-1] == 14.0
    assert ds.init_time.values == np.datetime64(grib_mirror.COMPLETE)
    assert ds.attrs["model_version"] == 2
    assert ds.tp.attrs["units"] == "mm"
    for param, offset in (("tp", 0.0), ("cp", 1000.0)):
        for member in range(3):
            for i, step in enumerate(grib_mirror.STEPS):
                base = offset + 100.0 * member + step
                got = ds[param].values[member, i]
                np.testing.assert_allclose(got[0], base + 0.01 * ROW_43_5, atol=1e-3)
                np.testing.assert_allclose(got[-1], base + 0.01 * ROW_47_5, atol=1e-3)


def test_run_of_the_first_model_version_has_no_convective_part(
    synthetic: DictConfig,
) -> None:
    ds = aifs_ens.fetch_run(synthetic, grib_mirror.VERSION_1, ALPS)
    assert list(ds.data_vars) == ["tp"]
    assert ds.attrs["model_version"] == 1


def test_second_mirror_is_used_when_the_first_lacks_the_run(
    synthetic: DictConfig,
) -> None:
    ds = aifs_ens.fetch_run(synthetic, grib_mirror.FALLBACK, ALPS)
    assert ds.init_time.values == np.datetime64(grib_mirror.FALLBACK)
    assert ds.tp.shape == (3, 2, 17, 33)


def test_run_on_no_mirror_is_reported_as_not_available(synthetic: DictConfig) -> None:
    with pytest.raises(aifs_ens.RunNotAvailable) as excinfo:
        aifs_ens.fetch_run(synthetic, grib_mirror.NOWHERE, ALPS)
    assert "2026-06-03" in str(excinfo.value)
    assert "mirrors" not in str(excinfo.value)


def test_run_with_a_member_missing_from_the_index_is_refused(
    synthetic: DictConfig,
) -> None:
    with pytest.raises(ValueError, match="tp"):
        aifs_ens.fetch_run(synthetic, grib_mirror.INCOMPLETE, ALPS)


@pytest.mark.parametrize(
    "other",
    [{"param": "cp", "number": "1"}, {"param": "tp", "number": "2"}],
    ids=["another parameter", "another member"],
)
def test_index_entry_that_points_at_another_message_is_refused(
    synthetic: DictConfig, tmp_path: Path, other: dict[str, str]
) -> None:
    mirror = tmp_path / "mirror"
    grib_mirror.write_run(mirror, grib_mirror.COMPLETE, ("tp", "cp"), version=2)
    index = next(mirror.rglob("*-6h-enfo-pf.index"))
    entries = [json.loads(line) for line in index.read_text().splitlines()]
    wanted = [{"param": "tp", "number": "1"}, other]
    first, second = (
        next(entry for entry in entries if want.items() <= entry.items())
        for want in wanted
    )
    for key in ("_offset", "_length"):
        first[key], second[key] = second[key], first[key]
    index.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    OmegaConf.update(synthetic, "data.aifs_ens.base_urls", [mirror.as_posix()])

    with pytest.raises(ValueError, match="unexpected"):
        aifs_ens.fetch_run(synthetic, grib_mirror.COMPLETE, ALPS)


def test_deaccumulate_gives_totals_per_step() -> None:
    accumulated = xr.Dataset(
        {"tp": ("step", [2.0, 5.0, 5.0, 4.95])}, coords={"step": [6, 12, 18, 24]}
    )
    out = aifs_ens.deaccumulate(accumulated)
    # the last step loses 0.05 mm to packing precision and is clipped at zero
    np.testing.assert_allclose(out.tp.values, [2.0, 3.0, 0.0, 0.0])
    assert list(out.step.values) == [6, 12, 18, 24]


def test_deaccumulate_leaves_variables_without_steps_alone() -> None:
    accumulated = xr.Dataset(
        {"tp": ("step", [2.0, 5.0]), "model_version": ((), 2)},
        coords={"step": [6, 12]},
    )
    out = aifs_ens.deaccumulate(accumulated)
    assert out.model_version.dims == ()
    assert out.model_version.dtype == accumulated.model_version.dtype
    assert int(out.model_version) == 2
    np.testing.assert_allclose(out.tp.values, [2.0, 3.0])


def test_deaccumulate_needs_contiguous_steps_from_the_first_interval() -> None:
    gappy = xr.Dataset({"tp": ("step", [2.0, 5.0])}, coords={"step": [6, 18]})
    with pytest.raises(ValueError):
        aifs_ens.deaccumulate(gappy)
