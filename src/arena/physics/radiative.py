"""Brightness-temperature features from the geostationary mosaic."""

from typing import ClassVar

import numpy as np
import xarray as xr

from arena.physics.base import register

# Infrared window (cloud-top temperature) and upper-level water vapour, in K.
IR = "mosaic_11200_nm"
WV = "mosaic_6200_nm"


@register
class Radiative:
    """Window and water-vapour temperatures, their difference, and the
    hourly change of the window temperature.

    ``dbt_dt`` at time t is (BT(t) - BT(t - 1 h)) / 1 h: it uses no later
    frame, and is missing when the previous frame is not one hour earlier.
    Strongly negative values mark rapidly cooling, growing cloud tops.
    """

    name: ClassVar[str] = "radiative"
    requires: ClassVar[tuple[str, ...]] = (IR, WV)
    known_future: ClassVar[bool] = False

    def compute(self, ds: xr.Dataset) -> xr.Dataset:
        ir, wv = ds[IR], ds[WV]
        gap = ds.time.diff("time", label="upper") / np.timedelta64(1, "h")
        change = ir.diff("time", label="upper").where(gap == 1.0)
        return xr.Dataset(
            {
                "bt_ir": ir.assign_attrs(units="K"),
                "bt_wv": wv.assign_attrs(units="K"),
                "bt_diff": (ir - wv).assign_attrs(units="K"),
                "dbt_dt": change.reindex(time=ds.time).assign_attrs(units="K h-1"),
            }
        )
