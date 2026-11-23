"""Per-pixel covariates from the AIFS ensemble, known ahead of the issue time.

Members are regridded to the common grid before any statistic is taken,
because quantiles do not commute with area averaging.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import ClassVar

import xarray as xr
from omegaconf import DictConfig

from arena.data.align import nwp_as_of
from arena.data.grid import common_grid, conservative_regrid
from arena.data.region import Region
from arena.physics.base import covariates, register

# Below this ensemble-mean total (mm per 6 h) the convective share is set to 0.
_WET_MM = 0.1


@register
class AifsEns:
    """Ensemble mean, 10/50/90 % quantiles and spread of 6-hour ``tp``."""

    name: ClassVar[str] = "aifs_ens"
    requires: ClassVar[tuple[str, ...]] = ("tp",)
    known_future: ClassVar[bool] = True

    def compute(self, ds: xr.Dataset) -> xr.Dataset:
        tp = ds.tp
        q = tp.quantile([0.1, 0.5, 0.9], dim="member").drop_vars("quantile")
        return xr.Dataset(
            {
                "tp_mean": tp.mean("member"),
                "tp_q10": q.isel(quantile=0),
                "tp_q50": q.isel(quantile=1),
                "tp_q90": q.isel(quantile=2),
                "tp_spread": tp.std("member"),
            }
        )


@register
class AifsEnsConvective:
    """Convective share of the ensemble-mean total.

    Only model version 2 produces ``cp``, so this provider is not in the
    default set of the models trained on version 1 forecasts.
    """

    name: ClassVar[str] = "aifs_ens_cp"
    requires: ClassVar[tuple[str, ...]] = ("tp", "cp")
    known_future: ClassVar[bool] = True

    def compute(self, ds: xr.Dataset) -> xr.Dataset:
        tp, cp = ds.tp.mean("member"), ds.cp.mean("member")
        share = (cp / tp).clip(0.0, 1.0).where(tp >= _WET_MM, 0.0)
        return xr.Dataset({"cp_fraction": share})


def forecast_covariates(
    runs: xr.Dataset,
    issue: datetime,
    leads_hours: Sequence[int],
    cfg: DictConfig,
    names: Sequence[str],
) -> xr.DataArray:
    """Covariates of the newest usable run at ``issue``, on the common grid.

    ``runs`` holds per-step totals (see ``aifs_ens.deaccumulate``).
    """
    forecast = nwp_as_of(runs, issue, leads_hours, cfg)
    lat, lon = common_grid(Region.from_config(cfg.region), float(cfg.data.grid_step))
    fine = forecast.map(lambda var: conservative_regrid(var, lat, lon))
    return covariates(fine, names)
