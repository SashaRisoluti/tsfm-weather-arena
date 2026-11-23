"""Solar geometry and calendar harmonics, known for any future time."""

from typing import ClassVar

import numpy as np
import pandas as pd
import xarray as xr

from arena.physics.base import register


def cos_solar_zenith(
    time: np.ndarray, latitude: np.ndarray, longitude: np.ndarray
) -> np.ndarray:
    """Cosine of the solar zenith angle, shape (time, latitude, longitude).

    NOAA general solar position formulas (fractional-year Fourier series for
    the declination and the equation of time); accurate to a fraction of a
    degree, which is ample for a covariate. Times are UTC.
    """
    stamps = pd.DatetimeIndex(time)
    hour = (stamps.hour + stamps.minute / 60 + stamps.second / 3600).to_numpy()
    days = np.where(stamps.is_leap_year, 366, 365)
    gamma = 2 * np.pi / days * (stamps.day_of_year.to_numpy() - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    # True solar time in minutes, then the hour angle in radians.
    solar_minutes = (
        hour[:, None] * 60 + eqtime[:, None] + 4 * np.asarray(longitude)[None, :]
    )
    angle = np.radians(solar_minutes / 4 - 180)
    lat = np.radians(np.asarray(latitude))
    return (
        np.sin(lat)[None, :, None] * np.sin(decl)[:, None, None]
        + np.cos(lat)[None, :, None]
        * np.cos(decl)[:, None, None]
        * np.cos(angle)[:, None, :]
    )


@register
class Solar:
    """Cosine of the solar zenith angle, and day-of-year and UTC-hour harmonics."""

    name: ClassVar[str] = "solar"
    requires: ClassVar[tuple[str, ...]] = ()
    known_future: ClassVar[bool] = True

    def compute(self, ds: xr.Dataset) -> xr.Dataset:
        stamps = pd.DatetimeIndex(ds.time.values)
        day = 2 * np.pi * (stamps.day_of_year.to_numpy() - 1) / 365.25
        hour = 2 * np.pi * (stamps.hour + stamps.minute / 60).to_numpy() / 24
        cosz = cos_solar_zenith(ds.time.values, ds.latitude.values, ds.longitude.values)
        dims = ("time", "latitude", "longitude")
        return xr.Dataset(
            {
                "cos_zenith": (dims, cosz),
                "doy_sin": ("time", np.sin(day)),
                "doy_cos": ("time", np.cos(day)),
                "hour_sin": ("time", np.sin(hour)),
                "hour_cos": ("time", np.cos(hour)),
            },
            coords={
                "time": ds.time,
                "latitude": ds.latitude,
                "longitude": ds.longitude,
            },
        )
