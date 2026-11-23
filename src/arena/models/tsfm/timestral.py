"""Timestral (Mistral AI), run in its own environment.

The inference code is installed from a pinned commit into the separate
environment ``external/timestral`` and driven through ``bridge.py``: one
process per model, requests and replies exchanged as ``.npz`` files. Weights
are under Mistral's modified MIT licence (see the README).
"""

import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from arena.paths import project_root


class TimestralBackend:
    def __init__(
        self, repo_id: str, device: str = "cpu", command: list[str] | None = None
    ) -> None:
        if command is None:
            project = project_root() / "external" / "timestral"
            command = [
                "uv",
                "run",
                "--project",
                str(project),
                "--no-sync",
                "python",
                str(project / "bridge.py"),
            ]
        self._folder = tempfile.TemporaryDirectory(prefix="timestral-")
        self._process = subprocess.Popen(
            [*command, repo_id, device],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in self._lines():
            if line.startswith("ready "):
                _, levels, known = line.split()
                self.levels = tuple(float(level) for level in levels.split(","))
                self.known_covariates = known == "1"
                break
        self._count = 0

    def _lines(self):  # type: ignore[no-untyped-def]
        assert self._process.stdout is not None
        for line in self._process.stdout:
            yield line.strip()
        raise RuntimeError(
            f"the Timestral bridge stopped (exit {self._process.wait()})"
        )

    def forecast(
        self,
        context: np.ndarray,
        past: np.ndarray | None,
        known: np.ndarray | None,
        horizon: int,
    ) -> tuple[np.ndarray, Sequence[float]]:
        self._count += 1
        folder = Path(self._folder.name)
        request = folder / f"request-{self._count}.npz"
        reply = folder / f"reply-{self._count}.npz"
        arrays = {"context": context, "horizon": np.int64(horizon)}
        if past is not None:
            arrays["past_covariates"] = past
        if known is not None:
            arrays["known_covariates"] = known
        np.savez(request, **arrays)  # type: ignore[arg-type]
        assert self._process.stdin is not None
        self._process.stdin.write(f"{request} {reply}\n")
        self._process.stdin.flush()
        answer = next(self._lines())
        request.unlink()
        if answer != "done":
            raise RuntimeError(f"Timestral failed: {answer.removeprefix('error ')}")
        with np.load(reply) as data:
            quantiles = data["quantiles"]
        reply.unlink()
        return quantiles, self.levels

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        self._process.wait(timeout=60)
        self._folder.cleanup()
