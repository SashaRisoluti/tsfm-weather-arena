"""Terrain strata used to sample and to report by type of location."""

import numpy as np
import xarray as xr
from omegaconf import DictConfig

STRATA = ("sea", "coast", "plains", "pre_alps", "alps")


def strata(elevation: xr.DataArray, cfg: DictConfig) -> xr.DataArray:
    """Label every grid cell with the index of its stratum in ``STRATA``.

    Land is split by elevation into plains, pre-Alps and Alps;
    lowland within ``coast_distance_deg`` of the sea is coast instead.
    """
    settings = cfg.data.strata
    # shortcut: sea is where the elevation model is exactly 0 m, its ocean
    # value; use a land-sea mask if a region has dry land at 0 m.
    sea = elevation == 0
    reach = round(float(settings.coast_distance_deg) / float(cfg.data.grid_step))
    window = 2 * reach + 1
    near_sea = (
        sea.astype("float64")
        .rolling(latitude=window, longitude=window, center=True, min_periods=1)
        .max()
        > 0
    )
    lowland = elevation < float(settings.lowland_max_m)
    labels = xr.full_like(elevation, STRATA.index("alps"), dtype="int8")
    labels = labels.where(
        elevation >= float(settings.foothill_max_m), STRATA.index("pre_alps")
    )
    labels = labels.where(~lowland, STRATA.index("plains"))
    labels = labels.where(~(lowland & near_sea), STRATA.index("coast"))
    labels = labels.where(~sea, STRATA.index("sea")).astype("int8")
    labels.attrs = {
        "long_name": "Terrain stratum",
        "flag_values": np.arange(len(STRATA), dtype="int8"),
        "flag_meanings": " ".join(STRATA),
    }
    return labels


def stratum_means(field: xr.DataArray, labels: xr.DataArray) -> xr.DataArray:
    """Mean of ``field`` over the cells of each stratum.

    The result has a leading ``stratum`` dimension labelled with the names in
    ``STRATA``; strata without any cell are left out.
    """
    parts = []
    for index, name in enumerate(STRATA):
        cells = labels == index
        if bool(cells.any()):
            mean = field.where(cells).mean(("latitude", "longitude"))
            parts.append(mean.assign_coords(stratum=name))
    return xr.concat(parts, dim="stratum")
