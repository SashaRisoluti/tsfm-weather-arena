"""Project-relative paths, so logs and notebook outputs never show absolute ones."""

import os
from pathlib import Path


def project_root() -> Path:
    """Return the nearest directory at or above the working directory that
    holds ``pyproject.toml``."""
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("no pyproject.toml at or above the working directory")


def rel(path: str | os.PathLike[str]) -> str:
    """Return ``path`` relative to the project root, with forward slashes.

    Raises ``ValueError`` for a path outside the project.
    """
    try:
        return Path(path).resolve().relative_to(project_root()).as_posix()
    except ValueError:
        # pathlib's own message quotes both absolute paths
        raise ValueError("path is outside the project root") from None
