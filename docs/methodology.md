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

The windows are fixed in the configuration before any model is trained.
Windows are defined on issue times, with inclusive dates. A sample is kept
only if its issue time is at least 48 hours before the end of its window, so
no target reaches into the next window (embargo). Tests check that the
windows of each family are ordered and disjoint, that the test window
overlaps no training or validation window, and that no kept sample has a
target outside its window.

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

## Alignment and covariates

### Common grid and regridding

All features live on the 0.1° grid of the target, 41 × 81 points over the
region. Only the forecasts need remapping. Their 0.25° totals are remapped
conservatively: each 0.1° cell takes the mean of the 0.25° cells it overlaps,
weighted by overlap area on the sphere. On regular grids the weights are a
product of a longitude factor (overlap in degrees) and a latitude factor
(overlap in sine of latitude), computed in `src/arena/data/grid.py`. Members
are remapped before ensemble statistics are taken, because quantiles do not
commute with averaging. The general regridding package `xesmf` was not used:
its ESMF backend installs only through conda.

An independent check splits every 0.25° cell into 50 × 50 sub-cells and
integrates precipitation times area directly. Over all 51 members and 10
steps of one run the two integrals agree to a relative 3 × 10⁻¹⁰
(notebook 02).

### Six-hour totals

In the short-range regime the target is a 6-hour total ending at 00, 06, 12
or 18 UTC, matching the forecast steps. It is the sum of the 24 frames at
times *s* with *T* − 6 h < *s* ≤ *T*, each standing for 15 minutes. A window
with any frame missing is missing.

### What is visible at an issue time

| Input | Usable at issue time *t* |
|---|---|
| Forecast run | newest run started at or before *t* − 12 h |
| Satellite frame | frames up to *t* − 60 min (measured latency about an hour) |
| Target history | frames up to *t* (hindcast: they are published 2 to 8 h later); up to *t* − 90 min in the sensitivity run |

Lead *L* of a forecast issued at *t* is read from step *t* + *L* − *r* of
run *r*. Runs at 00 and 12 UTC are archived to 60 h, so short-range issue
times are 00 and 12 UTC: an issue at 06 or 18 UTC would need steps to 66 h.
Tests sweep issue times and check that no join returns data from after its
delay (`tests/test_align.py`).

### Feature providers

Covariates come from feature providers (`src/arena/physics/base.py`), each
with a name, the input variables it needs, and a flag for features known
ahead of the issue time. A registry selects providers by name, and
`covariates` stacks their outputs along a `feature` dimension with that
flag. Models will see only this stack, so a new family of features is added by
registering a provider (and importing its module, or listing it among the
built-in ones); a test registers one at run time and checks that it reaches
the stack. Provider names must be unique.

| Provider | Features | Known ahead |
|---|---|---|
| `radiative` | 11.2 µm and 6.2 µm temperatures, their difference, change of the 11.2 µm temperature over the previous hour | no |
| `solar` | cosine of the solar zenith angle, day-of-year and UTC-hour harmonics | yes |
| `aifs_ens` | ensemble mean, 10, 50 and 90 % quantiles and spread of 6-hour `tp` | yes |
| `aifs_ens_cp` | convective share of the ensemble-mean total (0 below 0.1 mm) | yes |

The hourly change uses the current and the previous frame only, and is
missing across a gap. The solar zenith angle follows the NOAA solar position
formulas. `aifs_ens_cp` needs `cp`, which only model version 2 produces, so it
is left out of the default set of models trained on version 1 forecasts.

### Satellite cooling and rain onsets

Notebook 02 correlates the hourly cooling rate with rain onsets (an hour
with at least 1 mm/h after two hours below 0.1 mm/h) over June–August 2025.
In every stratum the correlation is −0.06 to −0.11 in the onset hour and the
hour before, weak two hours before, and indistinguishable from zero from
three hours on. The target at time *s* is itself estimated from satellite
frames between *s* − 120 min and *s* + 90 min, so the whole signal lies where
the covariates and the target share inputs. Any skill the satellite
covariates add at short leads is therefore judged together with the
90-minute sensitivity run.

## Evaluation

### Common quantiles

Every model is scored on the same 19 quantile levels, 0.05 to 0.95 in steps
of 0.05 (`src/arena/eval/quantiles.py`). Models that publish other levels are
mapped by linear interpolation in probability; beyond their outermost levels
the two nearest levels are extended linearly. The result is clipped at zero
and sorted. An ensemble becomes quantiles by numpy's default empirical
quantile (linear between order statistics). A point forecast has all 19
quantiles equal.

### Scores

| Score | Definition |
|---|---|
| CRPS | twice the mean pinball loss over the 19 levels, for every model; for the ensemble, the exact ensemble CRPS is reported beside it |
| CRPS skill | 1 − CRPS / CRPS of climatology |
| MAE | absolute error of the median |
| CSI, POD, FAR | event forecast where the median reaches the threshold; from hits, misses and false alarms summed over all samples |
| Brier score, reliability | probability of reaching the threshold, from the forecast CDF, linear between quantiles and flat outside them (at most 0.95, at least 0.05); reliability in 10 bins |
| Coverage | share of observations inside the 50 % and 90 % central intervals |
| PIT | forecast CDF at the observation; a dry observation gets a value drawn uniformly between 0 and the forecast's probability of zero |
| FSS | fractions skill score of two fields for a threshold and a square window |

Thresholds are 0.1, 1 and 5 mm/h for the 15-minute regime and the same rates
over 6 hours (0.6, 6 and 30 mm) for 6-hour totals. Every score has a unit
test against hand-computed values; the CRPS also against the `scoringrules`
package. Because most cell-hours are dry, coverage is high and Brier scores
are low for every model, so both are read together with the event scores.
The flat tails cap event probabilities at 0.05 and 0.95, which costs point
forecasts a little in the Brier score. On a spread distribution the CRPS
from 19 quantiles runs a few per cent above the exact value; for a point
forecast it is exact. A sample is scored only where both the observation and
every quantile of the forecast exist, and all models are scored on the same
samples.

Implemented and unit-tested, but not yet used by a scoring run: the exact
ensemble CRPS, PIT, reliability, FSS and the AIFS reference.

### Evaluation pixels

The grid is cut into blocks of 5 × 5 cells and a quarter of the blocks,
drawn once with a fixed seed, is held out. The 200 evaluation pixels are 40
per stratum drawn from held-out blocks (`configs/pixels/alps_eval.csv`);
models trained on pixels will use cells of the other blocks only. This keeps
most training cells away from evaluation cells, but not all: an evaluation
pixel on a block edge can border a training block (about half of the 200
do). Drawing from block interiors would leave too few coast cells.

### Baselines and reference

| Model | Forecast |
|---|---|
| Persistence | the last visible value, for every lead |
| Climatology | quantiles of the values at the same cell, calendar month and UTC hour of the valid time, fitted on 2016–2024 |
| Extrapolation | the last frame shifted along one motion vector: the shift, no faster than 120 km/h, that maximises the cross-correlation of the frames one hour apart; no growth or decay |
| AIFS ensemble | quantiles of the 51 members of the newest usable run, remapped conservatively to the target grid |

The extrapolation baseline is simpler than a full optical-flow nowcast (one
motion vector, whole-cell shifts), because the standard package for it builds
only from source. It sets a floor, not the state of the art.

A smoke run on three validation days (15–17 July 2025, hourly issues,
200 pixels, 14,400 samples per lead) checks the whole chain:
`arena eval smoke` writes `results/summary/smoke_regime_a.csv`. The days are
mostly dry and heavy-rain events are few, so the numbers test the code, not
the models. On CRPS, persistence
beats climatology only at 15 minutes; extrapolation beats persistence at 30
and 60 minutes but not at 15, 120 or 180. On CSI at 0.1 and 1 mm/h,
extrapolation is ahead of persistence at every lead. The
target looks up to 90 minutes ahead, which favours persistence at the
shortest leads; the sensitivity run (`lag_minutes` in the runner) removes
that advantage.

## Zero-shot foundation models

### Models

| Model | Weights | Quantiles | Covariates | Runs in |
|---|---|---|---|---|
| TimesFM 3 | `google/timesfm-3.0-pytorch`, about 300M parameters | 0.1 to 0.9 | past-only and known-future | the project environment (`uv sync --group tsfm`) |
| Timestral | `mistralai/Timestral-{5M,28M,123M,209M}` | 0.01 to 0.99 | past-only and known-future | its own environment, `external/timestral` |

Timestral is installed from a pinned commit of its repository and runs in a
separate process; the two exchange `.npz` files (`external/timestral/bridge.py`).
Neither model is trained here. Both forecasters share one wrapper
(`src/arena/models/tsfm/base.py`) that builds the input series, calls the
model and maps its quantiles to the 19 common levels, so the only
model-specific code is the call itself. TimesFM 3 is asked for non-negative
output; Timestral's quantiles are clipped at zero by the common mapping.

### Inputs and variants

| Regime | Target | Context | Leads | Issues |
|---|---|---|---|---|
| A | 15-minute rate, mm/h | 672 steps (7 days) | 15 min to 6 h, every 15 min | smoke: every 3 h on 15–17 July 2025 |
| B | 6-hour total, mm | 480 steps (120 days) | 6 h to 48 h, every 6 h | smoke: 00 and 12 UTC, July–August 2025 |

The context is the last values of the target visible at the issue time
(see above). Each run changes one thing against the plain run:

- **neighbours**: the pixel and its eight neighbouring cells as nine
  variates of one multivariate series (cells beyond the grid edge repeat the
  edge cell); only the centre is scored;
- **covariates** (regime A only): the `radiative` features as past-only
  covariates, each 15-minute step taking the newest hourly frame published by
  then (frames are published one hour after their time; a step whose newest
  frame was published more than an hour earlier is left missing), and the `solar`
  features as known-future covariates over context and horizon;
- **log1p**: the model sees log(1 + rate) and its quantiles are mapped back
  with exp − 1 (quantiles commute with a monotone transform);
- **size** (Timestral only): the 5M, 28M and 123M checkpoints beside 209M.

Run names and settings are in `configs/tsfm.yaml`; `arena run zero-shot
--regime a|b --split smoke|test` scores every run and the baselines on the
same issues and pixels. Scores are saved per model and issue day under
`results/parts/` (not committed), so a run that stops resumes where it
left off. Each part is written under a temporary name and renamed when
complete, and each model folder records the settings that produced it
(`settings.json`); a run with other settings refuses that folder instead of
mixing old and new parts. The summary covers only the configured models
that have every issue day scored, and fails if they were scored on
different numbers of samples. It is written to
`results/summary/zero_shot_<regime>_<split>.csv`. The models run on a CUDA
GPU when one is present; scores from a GPU and a CPU differ only by
floating-point rounding (about 1e-7 relative).

### Smoke results and caveats

The smoke run is read in `notebooks/03_zero_shot_benchmark.ipynb`. It uses
validation days, not the test window, and has no confidence intervals yet;
regime A covers only three mostly dry days. Two things in it need care:

- most samples are dry and a forecast whose quantiles are all zero counts a
  dry observation as covered, so 90 % coverage near 1 says little about
  calibration on wet samples;
- the foundation models lead on CRPS in the first hour but trail
  persistence and extrapolation on CSI from 45 minutes on, because their
  medians drift towards dry. CRPS and event scores are read together.
