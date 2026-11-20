from datetime import date

from arena.config import load_config


def test_default_region_is_alps() -> None:
    region = load_config().region
    assert region.name == "alps"
    assert (region.lon_min, region.lon_max) == (6.0, 14.0)
    assert (region.lat_min, region.lat_max) == (43.5, 47.5)


def test_override_changes_value() -> None:
    assert load_config(["region.lon_max=13.0"]).region.lon_max == 13.0


def test_aifs_steps_and_hours() -> None:
    aifs = load_config().data.aifs_ens
    assert list(aifs.steps) == [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
    assert list(aifs.init_hours) == [0, 12]
    assert aifs.availability_lag_hours == 12


def test_pardig_unit_factor() -> None:
    assert load_config().data.pardig.rate_to_mm_per_hour == 1000.0


def test_split_windows_are_ordered_and_disjoint() -> None:
    splits = load_config().data.splits
    test_start, test_end = (date.fromisoformat(d) for d in splits.test)
    assert (test_start, test_end) == (date(2026, 5, 13), date(2026, 11, 15))
    for family in ("tsfm", "hybrid"):
        train_start, train_end = (date.fromisoformat(d) for d in splits[family].train)
        val_start, val_end = (date.fromisoformat(d) for d in splits[family].validation)
        assert train_start < train_end < val_start < val_end < test_start < test_end
