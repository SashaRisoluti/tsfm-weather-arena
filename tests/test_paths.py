from pathlib import Path

import pytest

from arena.paths import project_root, rel


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    project = tmp_path / "project"
    (project / "notebooks").mkdir(parents=True)
    (project / "pyproject.toml").write_text("")
    monkeypatch.chdir(project)
    return project.resolve()


def test_project_root_found_from_nested_dir(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(root / "notebooks")
    assert project_root() == root


def test_project_root_raises_outside_a_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        project_root()


def test_rel_is_root_relative_posix(root: Path) -> None:
    out = rel(root / "data" / "cache" / "x.zarr")
    assert out == "data/cache/x.zarr"
    assert ":" not in out and "\\" not in out


def test_rel_resolves_relative_input_against_cwd(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(root / "notebooks")
    assert rel("../data/x.nc") == "data/x.nc"


def test_rel_of_root_is_dot(root: Path) -> None:
    assert rel(root) == "."


def test_rel_outside_root_raises(root: Path) -> None:
    with pytest.raises(ValueError):
        rel(root / ".." / "elsewhere" / "x.nc")


def test_rel_outside_root_error_names_no_path(root: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError) as excinfo:
        rel(root / ".." / "elsewhere" / "x.nc")
    assert tmp_path.name not in str(excinfo.value)


def test_project_path_resolves_against_the_root_and_keeps_absolute_paths(
    tmp_path: Path,
) -> None:
    from arena.paths import project_path, project_root

    assert project_path("configs/x.csv") == project_root() / "configs" / "x.csv"
    assert project_path(str(tmp_path / "y.csv")) == tmp_path / "y.csv"
