# Methodology

This document records what the data are, how they are read, and the choices
that shape every result. It grows with the project; sections on models and
evaluation are added as those parts are built.

## Data

All times are UTC. The study region is the box 6–14° E, 43.5–47.5° N
(`configs/region/alps.yaml`). The precipitation and satellite products share a
0.1° grid, on which the box is 41 × 81 points. Every path, variable name,
unit factor and date below is set in `configs/data.yaml`.

### Precipitation target: PARDIG

- **What.** PARDIG is a global precipitation estimate published by Google
  (Shalev et al., 2026, arXiv:2611.13941). The near-real-time product is used:
  one frame every 15 minutes from 2016-01-01 07:45 to the present.
- **Why only the near-real-time product.** A long-record product (1999–2023)
  exists but is built from different inputs. Using one product for both
  training and testing avoids a change of distribution between them.
- **Units.** Values are *instantaneous rates* in m/h, not accumulations. They
  are converted to mm/h once, when a regional subset is cached. An
  accumulation over a period is the mean rate over the frames of that period
  times its length. The product caps rates at 75 mm/h.
- **Layout.** Up to 2025 the region is cut from a store chunked as time
  series; from 2026 each frame is a separate global store. A frame is used
  only if its upload marker is present, and a frame that is entirely missing
  values counts as absent.

Three properties of the target limit what the results can claim.

1. **The target is itself a model output.** PARDIG is a learned estimate,
   trained against combined radar–radiometer retrievals from the GPM core
   satellite. Its inputs include geostationary imagery (the same family of
   data used here as covariates) and short-range forecast fields from the
   ECMWF physical model. Errors of the target are therefore not independent
   of the satellite covariates, and forecasts from ECMWF-family models may
   agree with it for reasons other than skill.
2. **The target looks ahead.** According to the product documentation, the
   estimate valid at time *s* uses satellite frames from *s* − 120 min to
   *s* + 90 min, and is published 2 to 8 hours after *s*. A model that is
   given the target history up to its issue time is thus given information a
   real-time system would not have. The benchmark is a hindcast. For the
   nowcasting regime, the evaluation protocol includes a sensitivity run that
   cuts the target history 90 minutes before the issue time; its results will
   be reported next to the main ones.
3. **The target is probably too dry in high terrain.** Over 2016–2025 the
   product gives less precipitation above 1000 m than over the plains (about
   670 mm against 1000 mm a year in this region). Mountains normally receive
   more precipitation than the plains next to them, so this is read as an
   underestimate. It was not checked here against gauge or radar data. The
   product paper reports that at some high-elevation stations, in the Alps
   among other ranges, its skill falls below that of a gauge-adjusted
   satellite product, and that it gives too little winter precipitation
   relative to summer over the European Alps; it names the limited
   sensitivity of the satellite radar to light snowfall as a possible cause.
   Scores in the Alps stratum measure agreement with this product rather
   than with the ground.

### Satellite covariates: SatWeave

- **What.** SatWeave is a calibrated mosaic of five geostationary satellites
  on the same 0.1° grid (stored north to south), with 17 spectral bands:
  brightness temperatures in K and reflectances in %.
- **Bands used.** `mosaic_11200_nm` (11.2 µm infrared window) and
  `mosaic_6200_nm` (6.2 µm upper-level water vapour). Over Europe only the
  bands carried by Meteosat hold data; the 10.4 µm window band does not.
- **Cadence.** The 2016–2025 store has 30-minute steps and later frames are
  hourly. Hourly steps are used throughout, so that covariates have the same
  cadence in training and in testing.

### Forecasts: AIFS ENS open data

- **What.** The AIFS ensemble is ECMWF's machine-learning ensemble forecast:
  51 members, four runs a day, 6-hour steps to 15 days, distributed as open
  data on a 0.25° grid (17 × 33 points over the region).
- **Archive.** Public mirrors keep past runs from the first operational one,
  2025-07-01 06 UTC. A scan on 2026-11-20 found all 1,014 runs at 00 and
  12 UTC; the other runs were not checked. Only the precipitation messages
  of each run are downloaded, by byte range, and cut to the region. Each
  message is checked against its index entry (parameter, member, step,
  units). Runs at 00 and 12 UTC are archived, for steps 6 to 60 h.
- **Variables.** Total precipitation `tp` and convective precipitation `cp`,
  in mm accumulated from the start of the run. Totals per 6-hour step are
  differences between consecutive steps; small negative differences, which
  arise because each message is packed separately, are set to zero.
- **Model versions.** Runs up to 2026-05-12 00 UTC come from version 1 and
  later runs from version 2. The version of every archived run is read from
  the GRIB header (`generatingProcessIdentifier`) and kept with it. `cp`
  exists only from version 2. From 2026-10-20 12 UTC the published `tp` is
  packed with 12 bits instead of 16, so its resolution is 16 times coarser
  for the same range of values.
- **Availability.** ECMWF's schedule puts dissemination about 6 h 40 min to
  7 h 40 min after the nominal run time, and later arrivals were observed on
  the mirrors. A run is treated as usable 12 hours after its nominal time. At
  an issue time *t* a model may therefore use the run started at *t* − 12 h at
  the latest, which is why forecasts are archived to 60 h for lead times up
  to 48 h.

### Elevation and terrain strata

Terrain height comes from the Copernicus DEM GLO-90, read through the
Open-Meteo elevation API at the centre of each grid cell. Sea is where the
elevation model gives exactly 0 m, its value over the sea; lagoons and any land
at exactly 0 m fall in the same class. The other cells are grouped into coast
(land below 300 m within 0.3° of the sea), plains (other land below 300 m),
pre-Alps (300 to 1000 m) and Alps (1000 m and above). A point
sample is a noisy summary of a 10 km cell in steep terrain, so cells near a
threshold may be assigned to either side.

### Splits

The windows are fixed in the configuration before any model is trained; the
code that applies them is added with the models and the evaluation. Windows
are defined on issue times, with inclusive dates. A sample whose target would
reach into the next window is dropped (48-hour embargo).

| Models | Training | Validation | Test |
|---|---|---|---|
| Time-series foundation models and baselines | 2016-01-01 – 2024-12-31 | 2025-01-01 – 2025-12-31 | 2026-05-13 – 2026-11-15 |
| Models that use forecast covariates | 2025-07-02 – 2026-03-31 | 2026-04-01 – 2026-05-12 | 2026-05-13 – 2026-11-15 |

All models will be scored on the same test window and the same locations. The
test window lies entirely in forecast model version 2. Models that use
forecast covariates can only be trained where forecasts exist, so their
training and validation data come from version 1 and do not include `cp`;
the size of that mismatch is part of what is measured.

### Local cache

Regional subsets are kept under `data/cache/<region>/`, one store per unit of
work: the precipitation archive, one day of frames, one forecast run. A store
is written to a temporary folder and renamed, so a store that exists is
complete. Two writers of the same store do not disturb each other: the first
to finish wins. A day with missing frames is not cached unless gaps are
explicitly allowed, in which case the missing steps are counted. A step is
missing as soon as one of its variables has no data. A read that fails is
never recorded as a gap: the day or run is left for the next run of the
command.
