"""Command-line entry point: ``arena data ...`` and ``arena archive ...``."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from arena.config import load_config
from arena.data import aifs_archive, cache
from arena.paths import rel

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True, help="Fill the regional data cache.")
archive_app = typer.Typer(no_args_is_help=True, help="Archive ensemble forecast runs.")
app.add_typer(data_app, name="data")
app.add_typer(archive_app, name="archive")

Overrides = Annotated[
    list[str] | None,
    typer.Option("--override", "-o", help="Config override as key=value; repeatable."),
]
Start = Annotated[datetime, typer.Option(help="First day or run time, UTC.")]
End = Annotated[datetime, typer.Option(help="Last day or run time, UTC, inclusive.")]


@app.callback()
def main() -> None:
    """Precipitation forecasting benchmark."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def _report(paths: list[Path]) -> None:
    for path in paths:
        try:
            typer.echo(rel(path))
        except ValueError:  # cache kept outside the project
            typer.echo(path.name)
    typer.echo(f"{len(paths)} stores written")


@data_app.command("pardig-archive")
def pardig_archive(override: Overrides = None) -> None:
    """Cut the region out of the 2016-2025 precipitation archive."""
    _report([cache.cache_pardig_archive(load_config(override or []))])


@data_app.command("days")
def days(
    source: Annotated[str, typer.Argument(help="pardig or satweave.")],
    start: Start,
    end: End,
    allow_gaps: Annotated[
        bool, typer.Option(help="Cache days with missing frames, as NaN.")
    ] = False,
    override: Overrides = None,
) -> None:
    """Cache whole days of frames; days already cached are skipped."""
    cfg = load_config(override or [])
    try:
        written = cache.cache_days(cfg, source, start.date(), end.date(), allow_gaps)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None
    _report(written)


@data_app.command("elevation")
def elevation(override: Overrides = None) -> None:
    """Fetch the terrain elevation of the region."""
    _report([cache.cache_elevation(load_config(override or []))])


@archive_app.command("backfill")
def backfill(start: Start, end: End, override: Overrides = None) -> None:
    """Cache the forecast runs initialised between two times."""
    _report(aifs_archive.backfill(load_config(override or []), start, end))


@archive_app.command("update")
def update(override: Overrides = None) -> None:
    """Cache the runs disseminated since the newest cached one."""
    _report(aifs_archive.update(load_config(override or [])))


@archive_app.command("coverage")
def coverage(start: Start, end: End, override: Overrides = None) -> None:
    """Scan the mirrors and list which runs exist."""
    _report([aifs_archive.coverage(load_config(override or []), start, end)])
