from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from arena.config import load_config
from arena.data.grid import common_grid
from arena.data.region import Region
from arena.nwp.covariates import forecast_covariates
from arena.physics import base

LAT = np.array([45.0, 45.25])
LON = np.array([7.0, 7.25, 7.5])


def ensemble(tp: list[float], cp: list[float]) -> xr.Dataset:
    """Members along the first axis, the same value at every cell."""
    shape = (len(tp), 1, LAT.size, LON.size)

    def cube(values: list[float]) -> np.ndarray:
        return np.broadcast_to(np.array(values)[:, None, None, None], shape).copy()

    dims = ("member", "lead", "latitude", "longitude")
    return xr.Dataset(
        {"tp": (dims, cube(tp)), "cp": (dims, cube(cp))},
        coords={
            "member": np.arange(len(tp)),
            "lead": [6],
            "latitude": LAT,
            "longitude": LON,
        },
    )


def test_ensemble_statistics_by_hand() -> None:
    members = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    out = base.covariates(ensemble(members, [0.0] * 11), ["aifs_ens"])
    point = out.isel(lead=0, latitude=0, longitude=0)
    got = {str(name): float(point.sel(feature=name)) for name in out.feature.values}
    assert got["tp_mean"] == 5.0
    assert (got["tp_q10"], got["tp_q50"], got["tp_q90"]) == (1.0, 5.0, 9.0)
    assert np.isclose(got["tp_spread"], np.std(members))
    assert bool(out.known_future.all())
    assert out.dims == ("feature", "lead", "latitude", "longitude")


def test_convective_fraction_is_zero_where_the_forecast_is_dry() -> None:
    wet = base.covariates(ensemble([2.0, 4.0], [1.0, 2.0]), ["aifs_ens_cp"])
    dry = base.covariates(ensemble([0.0, 0.1], [0.0, 0.1]), ["aifs_ens_cp"])
    np.testing.assert_allclose(wet.sel(feature="cp_fraction").values, 0.5)
    np.testing.assert_allclose(dry.sel(feature="cp_fraction").values, 0.0)


def test_forecast_covariates_on_the_common_grid() -> None:
    cfg = load_config()
    region = Region.from_config(cfg.region)
    lat = np.round(43.5 + 0.25 * np.arange(17), 6)
    lon = np.round(6.0 + 0.25 * np.arange(33), 6)
    inits = pd.date_range("2026-06-01T00:00", periods=2, freq="12h")
    steps = np.arange(6, 61, 6)
    shape = (2, 3, steps.size, lat.size, lon.size)
    # 1 mm per step for member 0, 2 for member 1, 3 for member 2
    tp = np.broadcast_to(np.arange(1.0, 4.0)[None, :, None, None, None], shape)
    runs = xr.Dataset(
        {"tp": (("init_time", "member", "step", "latitude", "longitude"), tp.copy())},
        coords={
            "init_time": inits,
            "member": [0, 1, 2],
            "step": steps,
            "latitude": lat,
            "longitude": lon,
        },
    )
    out = forecast_covariates(runs, datetime(2026, 6, 2, 0), [6, 48], cfg, ["aifs_ens"])
    fine_lat, fine_lon = common_grid(region, cfg.data.grid_step)
    assert out.sizes == {
        "feature": 5,
        "lead": 2,
        "latitude": fine_lat.size,
        "longitude": fine_lon.size,
    }
    np.testing.assert_allclose(out.sel(feature="tp_mean").values, 2.0)
    assert out.init_time.values == np.datetime64("2026-06-01T12:00")
