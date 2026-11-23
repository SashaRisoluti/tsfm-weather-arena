# tsfm-weather-arena

Precipitation forecasting over Northern Italy and the Alps, set up as a
controlled comparison between three families of models:

1. **Time-series foundation models**, which see only the history of a point
   plus covariates.
2. **An open-source AI weather model**, which forecasts the global state of
   the atmosphere.
3. **A hybrid**, in which a time-series foundation model receives the weather
   model's forecasts as known-future covariates and corrects them locally.

## Research questions

- **Crossover.** Up to what lead time can a time-series foundation model
  compete, and where do the error curves of the three families cross?
- **Hybrid value.** How much does the hybrid improve on each of its two parts,
  and at which lead times?
- **Adaptation.** Which adaptation method gives the best skill per unit of
  cost (trainable parameters, GPU hours, amount of training data)?

Every comparison is reported with confidence intervals and significance
tests. A difference that is not significant is reported as no difference.

## Status

The data layer is in place: tested readers for the three sources, a
resumable regional cache, an incremental archive of forecast runs, and a
first look at the data in
[notebooks/01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb).
On top of it: a common grid with conservative regridding, joins that respect
a stated publication delay for each input, split windows with an embargo, and
covariate providers for satellite, solar and forecast features, examined in
[notebooks/02_alignment_and_physical_covariates.ipynb](notebooks/02_alignment_and_physical_covariates.ipynb).
The evaluation framework puts every model on the same 19 quantile levels and
scores it on a fixed set of held-out pixels. Baselines are persistence,
climatology and an extrapolation nowcast; stored AIFS ensemble forecasts can
be served as a reference model. `arena eval smoke` runs a quick check of the
baselines.
Units, model versions, splits and known caveats of the data are in
[docs/methodology.md](docs/methodology.md).

## Data

| Source | Role | Access | Licence |
|---|---|---|---|
| PARDIG, near-real-time product | Precipitation target: 15-minute rate, 0.1° | public bucket `gs://pardig_1` | CC BY 4.0 |
| SatWeave | Satellite covariates: hourly brightness temperatures, 0.1° | public bucket `gs://satweave_1` | CC BY 4.0 |
| ECMWF open data, AIFS ensemble | Forecasts: 51 members, 6-hour steps, 0.25° | public mirrors on Google Cloud and AWS | CC BY 4.0 |
| Copernicus DEM GLO-90 | Terrain elevation | Open-Meteo elevation API | see provider terms |

All sources are read anonymously over HTTPS; no account or key is needed.

### Getting the data

The commands below fill a local cache under `data/cache/` with the study
region only. They skip what is already cached, so they can be interrupted and
run again. A day or a run whose download fails is skipped with a warning and
fetched the next time the command runs.

```bash
uv run arena data pardig-archive                                       # 2016-2025 target
uv run arena data days pardig   --start 2026-01-01 --end 2026-11-17   # later target frames
uv run arena data days satweave --start 2025-01-01 --end 2026-11-17   # satellite bands
uv run arena data elevation
uv run arena archive backfill --start 2025-07-01 --end 2026-11-17     # forecast runs
uv run arena archive update                                            # runs still missing
```

The satellite and forecast files are global, so far more is transferred than
is kept. Measured on connections between about 2 and 15 MB/s:

| Command | Transfer | Time | Kept |
|---|---|---|---|
| `data pardig-archive` | 0.3 GB | under a minute | 0.3 GB |
| `data days pardig`, per day | 0.1 GB | 30 s | 0.1 MB |
| `data days satweave`, per day | 0.5 GB | 30 s to 2 min | 0.5 MB |
| `archive backfill`, per run | 0.5 to 0.9 GB | 40 to 60 s | 1 MB |

A full back-fill moves about 1 TB and takes most of a day.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Tests run offline on small synthetic stores shaped like the real ones.

### Foundation models

The foundation models are an optional dependency group, and Timestral runs in
its own environment:

```bash
uv sync --group tsfm
uv sync --project external/timestral
uv run arena run zero-shot --regime a --split smoke
```

They use a CUDA GPU when one is present (`tsfm.device=cpu` forces the CPU).
For a free GPU, [notebooks/colab_zero_shot.ipynb](notebooks/colab_zero_shot.ipynb)
runs the same command on Colab with the score parts kept on Google Drive.

## Licence and attribution

Code is released under the MIT licence (see [LICENSE](LICENSE)). Data sources
and model weights keep their own licences.

- **PARDIG.** Shalev et al. (2026), *PARDIG: Global Precipitation Estimates
  Learned from Sparse Satellite Observations*, arXiv:2611.13941. © 2026 Google
  LLC's machine learning models used to create the experimental data made
  available at
  https://developers.google.com/earth-engine/datasets/catalog/projects_gcp-public-data-weathernext_assets_PARDIG_1_NRT
  under CC BY 4.0 licence terms. This data is intended for experimental
  modelling only and is not intended, validated, or approved for real world
  use.
- **SatWeave.** Google DeepMind and Google Research, CC BY 4.0. Built from
  observations of the GOES, Himawari and Meteosat satellites.
- **AIFS ensemble forecasts.** ECMWF open data, © European Centre for
  Medium-Range Weather Forecasts, CC BY 4.0.
- **Elevation.** Copernicus DEM GLO-90, accessed through the Open-Meteo
  elevation API.

## Roadmap

A second module on cloud-top physics and convective initiation: cloud-top
height from satellite brightness temperature, detection of rapid cloud-top
cooling, and 3D rendering of clouds.
