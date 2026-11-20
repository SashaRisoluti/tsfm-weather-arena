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

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

## Licence

Code is released under the MIT licence (see [LICENSE](LICENSE)). Data sources
and model weights keep their own licences; attributions are listed here as
each source is integrated.

## Roadmap

A second module on cloud-top physics and convective initiation: cloud-top
height from satellite brightness temperature, detection of rapid cloud-top
cooling, and 3D rendering of clouds.
