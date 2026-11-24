"""Run Timestral in its own environment, driven through files.

Started once per model: ``python bridge.py <repo id> <device>``. It prints
``ready <levels> <known>`` when the model is loaded (quantile levels joined by
commas; ``known`` is 1 if the checkpoint accepts known-future covariates).
Then, for every line ``<request.npz> <reply.npz>`` read from stdin, it
forecasts and writes the reply, and prints ``done`` (or ``error <message>``).

A request holds ``context`` (B, V, T), ``horizon`` and optionally
``past_covariates`` (B, C, T) and ``known_covariates`` (B, C, T + horizon).
A reply holds ``quantiles`` (B, V, horizon, Q).
"""

import sys

import numpy as np
import torch
from timestral import TimestralForecaster


def main() -> None:
    repo_id, device = sys.argv[1], sys.argv[2]
    model = TimestralForecaster.from_pretrained(repo_id, device=device)
    levels = ",".join(f"{level:g}" for level in model.quantile_levels)
    print(f"ready {levels} {int(model.supports_known_covariates)}", flush=True)
    for line in sys.stdin:
        try:
            request_path, reply_path = line.split()
            with np.load(request_path) as request:
                kwargs = {
                    name: request[name]
                    for name in ("past_covariates", "known_covariates")
                    if name in request.files
                }
                with torch.inference_mode():
                    forecast = model.forecast(
                        request["context"],
                        int(request["horizon"]),
                        symmetric_averaging=True,
                        **kwargs,
                    )
            np.savez(reply_path, quantiles=forecast.quantiles.cpu().numpy())
            print("done", flush=True)
        except Exception as error:  # reported to the caller, which raises
            message = str(error).replace("\n", " ")
            print(f"error {type(error).__name__}: {message}", flush=True)


if __name__ == "__main__":
    main()
