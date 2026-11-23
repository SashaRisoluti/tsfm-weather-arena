"""Feature providers and the covariate tensor they feed.

A provider turns input variables on the common grid into named features.
Models and evaluation see only the tensor built by ``covariates``, so a new
family of features (for example cloud-top height) is added by registering a
provider, without touching them.
"""

import importlib
from collections.abc import Sequence
from typing import ClassVar, Protocol

import xarray as xr


class FeatureProvider(Protocol):
    name: ClassVar[str]
    # Variables the input dataset must hold.
    requires: ClassVar[tuple[str, ...]]
    # True for features known ahead of the issue time (solar geometry, forecasts).
    known_future: ClassVar[bool]

    def compute(self, ds: xr.Dataset) -> xr.Dataset: ...


PROVIDERS: dict[str, type[FeatureProvider]] = {}

# Modules whose providers ship with the package, imported on first lookup.
_BUILT_IN = (
    "arena.physics.radiative",
    "arena.physics.solar",
    "arena.nwp.covariates",
)


def register(cls: type[FeatureProvider]) -> type[FeatureProvider]:
    """Class decorator that makes a provider selectable by its name.

    Raises ``ValueError`` if another provider already has that name.
    """
    taken = PROVIDERS.get(cls.name)
    if taken is not None and taken is not cls:
        raise ValueError(f"a feature provider named {cls.name!r} already exists")
    PROVIDERS[cls.name] = cls
    return cls


def provider_names() -> list[str]:
    for module in _BUILT_IN:
        importlib.import_module(module)
    return sorted(PROVIDERS)


def covariates(ds: xr.Dataset, names: Sequence[str]) -> xr.DataArray:
    """Features of the named providers, stacked along a ``feature`` dimension.

    The ``known_future`` coordinate flags features known ahead of the issue
    time. Raises ``KeyError`` for an unknown provider and ``ValueError`` if
    ``ds`` lacks an input a provider requires.
    """
    provider_names()
    parts, flags = [], []
    for name in names:
        if name not in PROVIDERS:
            raise KeyError(f"no feature provider named {name!r}")
        provider = PROVIDERS[name]()
        missing = [var for var in provider.requires if var not in ds]
        if missing:
            raise ValueError(f"provider {name!r} needs {missing}")
        features = provider.compute(ds)
        parts.append(features)
        flags += [provider.known_future] * len(features.data_vars)
    stacked = xr.merge(parts, join="exact", compat="no_conflicts").to_dataarray(
        "feature"
    )
    return stacked.assign_coords(known_future=("feature", flags))
