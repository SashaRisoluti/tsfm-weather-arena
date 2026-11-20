"""Composed project configuration."""

from collections.abc import Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from arena.paths import project_root


def load_config(overrides: Sequence[str] = ()) -> DictConfig:
    """Compose ``configs/config.yaml``, applying ``key=value`` overrides."""
    config_dir = str(project_root() / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="config", overrides=list(overrides))
