from omegaconf import DictConfig
from typer.testing import CliRunner

from arena.cli import app
from arena.data import cache

runner = CliRunner()


def overrides(cfg: DictConfig) -> list[str]:
    """The synthetic configuration, spelled as command-line overrides."""
    quoted = {
        "data.cache_dir": cfg.data.cache_dir,
        "data.pardig.archive_url": cfg.data.pardig.archive_url,
        "data.pardig.frame_url": cfg.data.pardig.frame_url,
        "data.satweave.archive_url": cfg.data.satweave.archive_url,
        "data.satweave.frame_url": cfg.data.satweave.frame_url,
    }
    mirrors = ",".join(f'"{url}"' for url in cfg.data.aifs_ens.base_urls)
    values = [f'{key}="{value}"' for key, value in quoted.items()]
    values += [
        f"data.aifs_ens.base_urls=[{mirrors}]",
        f"data.aifs_ens.members={cfg.data.aifs_ens.members}",
        f"data.aifs_ens.steps={list(cfg.data.aifs_ens.steps)}",
    ]
    return [arg for value in values for arg in ("-o", value)]


def test_help_lists_both_command_groups() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "data" in result.output and "archive" in result.output


def test_data_days_caches_a_day(synthetic: DictConfig) -> None:
    args = ["data", "days", "pardig", "--start", "2026-01-01", "--end", "2026-01-01"]
    result = runner.invoke(app, args + overrides(synthetic))
    assert result.exit_code == 0, result.output
    assert "20260101.zarr" in result.output
    store = cache.cache_root(synthetic) / "pardig" / "frames" / "20260101.zarr"
    assert store.is_dir()


def test_data_days_reports_days_it_could_not_cache(synthetic: DictConfig) -> None:
    args = ["data", "days", "pardig", "--start", "2026-01-02", "--end", "2026-01-02"]
    result = runner.invoke(app, args + overrides(synthetic))
    assert result.exit_code == 0, result.output
    assert "0 stores written" in result.output


def test_unknown_source_is_a_usage_error(synthetic: DictConfig) -> None:
    args = ["data", "days", "radar", "--start", "2026-01-01", "--end", "2026-01-01"]
    result = runner.invoke(app, args + overrides(synthetic))
    assert result.exit_code == 2
    assert "unknown source" in result.output


def test_archive_backfill_caches_a_run(synthetic: DictConfig) -> None:
    args = ["archive", "backfill", "--start", "2026-06-01", "--end", "2026-06-01"]
    result = runner.invoke(app, args + overrides(synthetic))
    assert result.exit_code == 0, result.output
    assert (cache.cache_root(synthetic) / "aifs_ens" / "2026060100.zarr").is_dir()
    assert "1 stores written" in result.output
