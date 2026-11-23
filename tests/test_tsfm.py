import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from arena.eval.quantiles import LEVELS
from arena.finetune.windows import inverse, pixel_series, to_steps, transform
from arena.models.base import Forecaster
from arena.models.tsfm.base import TsfmForecaster
from arena.models.tsfm.timesfm3 import TimesFM3Backend
from arena.models.tsfm.timestral import TimestralBackend

LAT = np.round(45.0 + 0.1 * np.arange(4), 6)
LON = np.round(7.0 + 0.1 * np.arange(5), 6)
ISSUE = datetime(2025, 7, 15, 12, 0)
PIXELS = pd.DataFrame({"latitude": [45.0, 45.2], "longitude": [7.0, 7.3]})


def grid(times: pd.DatetimeIndex) -> xr.DataArray:
    """Values that name their place: 100 x row + column + 1000 x step."""
    rows, cols = np.indices((LAT.size, LON.size))
    values = 1000.0 * np.arange(times.size)[:, None, None] + 100 * rows + cols
    return xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": LAT, "longitude": LON},
    )


def context_times(steps: int = 8) -> pd.DatetimeIndex:
    return pd.date_range(end=ISSUE, periods=steps, freq="15min")


def test_pixel_series_centre_first_and_edges_clamped() -> None:
    series = pixel_series(grid(context_times(2)), PIXELS, neighbours=True)
    assert series.shape == (2, 9, 2)
    # pixel 0 sits in the corner (row 0, column 0): neighbours outside repeat it
    np.testing.assert_allclose(series[0, :, 0], [0, 0, 0, 1, 0, 1, 100, 100, 101])
    # pixel 1 is at row 2, column 3
    np.testing.assert_allclose(
        series[1, :, 1], 1000 + np.array([203, 102, 103, 104, 202, 204, 302, 303, 304])
    )
    single = pixel_series(grid(context_times(2)), PIXELS, neighbours=False)
    np.testing.assert_allclose(single[:, 0, 0], [0, 203])


def test_to_steps_is_an_as_of_join_with_a_lag() -> None:
    hourly = pd.date_range("2025-07-15T09:00", "2025-07-15T12:00", freq="1h")
    features = xr.DataArray(
        np.arange(4.0), dims="time", coords={"time": hourly}
    ).expand_dims(feature=["bt_ir"])
    steps = pd.date_range("2025-07-15T10:45", "2025-07-15T12:00", freq="15min")
    out = to_steps(features, steps, lag_minutes=60)
    # at 10:45 the 09:00 frame is the newest published; at 12:00 the 11:00 frame
    np.testing.assert_allclose(out.isel(feature=0).values, [0, 1, 1, 1, 1, 2])
    late = to_steps(features, pd.DatetimeIndex(["2025-07-15T15:30"]), lag_minutes=60)
    assert np.isnan(late.values).all()


def test_log1p_transform_round_trips() -> None:
    x = np.array([0.0, 0.5, 30.0])
    np.testing.assert_allclose(inverse(transform(x, "log1p"), "log1p"), x)
    np.testing.assert_allclose(transform(x, "none"), x)


class FakeBackend:
    """Quantile q of every series: the last context value of variate 0 + q,
    so the output shows which series and transform reached the backend."""

    levels = (0.1, 0.5, 0.9)

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def forecast(self, context, past, known, horizon):  # type: ignore[no-untyped-def]
        self.calls.append(
            dict(context=context, past=past, known=known, horizon=horizon)
        )
        last = context[:, :, -1]
        q = last[:, :, None, None] + np.array(self.levels)[None, None, None, :]
        return np.broadcast_to(q, (*last.shape, horizon, 3)).copy(), self.levels


def test_tsfm_forecaster_maps_the_backend_output_to_common_quantiles() -> None:
    backend = FakeBackend()
    model = TsfmForecaster("fake", backend, PIXELS, context_steps=4, neighbours=True)
    assert isinstance(model, Forecaster)
    context = grid(context_times()) / 1e4
    leads = [timedelta(minutes=15), timedelta(minutes=45)]
    out = model.predict(context, None, ISSUE, leads)
    assert out.dims == ("quantile", "lead", "pixel")
    np.testing.assert_allclose(out["quantile"].values, LEVELS)
    call = backend.calls[0]
    assert call["context"].shape == (2, 9, 4) and call["horizon"] == 3
    assert call["past"] is None and call["known"] is None
    last = context.isel(time=-1).values
    np.testing.assert_allclose(
        out.sel(quantile=0.5).isel(lead=1).values, [last[0, 0] + 0.5, last[2, 3] + 0.5]
    )


def test_log1p_is_applied_before_and_inverted_after() -> None:
    backend = FakeBackend()
    model = TsfmForecaster("fake", backend, PIXELS, context_steps=4, transform="log1p")
    context = xr.full_like(grid(context_times()), 3.0)
    out = model.predict(context, None, ISSUE, [timedelta(minutes=15)])
    np.testing.assert_allclose(backend.calls[0]["context"], np.log1p(3.0))
    np.testing.assert_allclose(
        out.sel(quantile=0.5).values, np.expm1(np.log1p(3.0) + 0.5)
    )


def test_past_covariates_never_reach_past_the_issue_time() -> None:
    backend = FakeBackend()
    model = TsfmForecaster(
        "fake", backend, PIXELS, context_steps=4, use_covariates=True
    )
    times = pd.date_range(end=ISSUE + timedelta(minutes=30), periods=10, freq="15min")
    values = grid(times)
    features = xr.concat([values, -values], dim="feature").assign_coords(
        feature=["bt_ir", "cos_zenith"], known_future=("feature", [False, True])
    )
    model.predict(grid(context_times()), features, ISSUE, [timedelta(minutes=30)])
    call = backend.calls[0]
    assert call["past"].shape == (2, 1, 4) and call["known"].shape == (2, 1, 6)
    newest_past = float(values.sel(time=ISSUE).isel(latitude=0, longitude=0))
    assert call["past"][0, 0, -1] == newest_past
    assert call["known"][0, 0, -1] == -float(
        values.isel(time=-1, latitude=0, longitude=0)
    )


def test_a_context_shorter_than_required_is_refused() -> None:
    model = TsfmForecaster("fake", FakeBackend(), PIXELS, context_steps=20)
    with pytest.raises(ValueError, match="steps of context"):
        model.predict(grid(context_times()), None, ISSUE, [timedelta(minutes=15)])


class FakeTimesFM3:
    """Stands in for TimesFM3Forecaster: quantiles 0.1 .. 0.9 around the last value."""

    class config:  # noqa: N801
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(self) -> None:
        self.kwargs: dict = {}

    def predict_batch(self, contexts, horizon, **kwargs):  # type: ignore[no-untyped-def]
        self.kwargs = kwargs
        for ctx in contexts:
            ctx = np.atleast_2d(ctx)
            q = ctx[:, -1][:, None, None] + np.linspace(-0.4, 0.4, 9)[None, None, :]
            q = np.broadcast_to(q, (ctx.shape[0], horizon, 9))
            yield type(
                "Out", (), {"quantiles": q[0] if np.ndim(contexts[0]) == 1 else q}
            )


def test_timesfm3_backend_shapes_and_arguments() -> None:
    fake = FakeTimesFM3()
    backend = TimesFM3Backend(model=fake)
    context = np.arange(2 * 3 * 5, dtype="float32").reshape(2, 3, 5)
    q, levels = backend.forecast(context, None, None, horizon=4)
    assert q.shape == (2, 3, 4, 9) and list(levels) == FakeTimesFM3.config.quantiles
    np.testing.assert_allclose(q[1, 2, 0, 4], context[1, 2, -1])
    assert fake.kwargs["return_quantiles"] is True
    univariate = np.arange(10, dtype="float32").reshape(2, 1, 5)
    past = np.zeros((2, 2, 5), "float32")
    q, _ = backend.forecast(univariate, past, None, horizon=4)
    assert q.shape == (2, 1, 4, 9)
    assert len(fake.kwargs["past_only_covariates"]) == 2


FAKE_BRIDGE = """
import sys
import numpy as np
print("noise before ready")
print("ready 0.1,0.5,0.9 0", flush=True)
for line in sys.stdin:
    request_path, reply_path = line.split()
    with np.load(request_path) as r:
        if "known_covariates" in r.files:
            print("error ValueError: no known covariates", flush=True)
            continue
        c, h = r["context"], int(r["horizon"])
        q = c[:, :, -1][:, :, None, None] + np.array([0.1, 0.5, 0.9])
        np.savez(reply_path, quantiles=np.broadcast_to(q, (*c.shape[:2], h, 3)))
    print("done", flush=True)
"""


def test_timestral_backend_talks_to_its_bridge(tmp_path: Path) -> None:
    script = tmp_path / "bridge.py"
    script.write_text(FAKE_BRIDGE)
    backend = TimestralBackend(
        "any/model", "cpu", command=[sys.executable, str(script)]
    )
    try:
        context = np.arange(12, dtype="float32").reshape(2, 1, 6)
        q, levels = backend.forecast(context, None, None, horizon=3)
        assert levels == (0.1, 0.5, 0.9)
        assert q.shape == (2, 1, 3, 3)
        np.testing.assert_allclose(q[1, 0, 2, 1], 11.5)
        assert not backend.known_covariates
        with pytest.raises(RuntimeError, match="no known covariates"):
            backend.forecast(context, None, np.zeros((2, 1, 9), "float32"), horizon=3)
    finally:
        backend.close()


def test_timesfm3_backend_passes_the_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict = {}

    class Loader:
        @staticmethod
        def from_pretrained(checkpoint, device=None, **kwargs):  # type: ignore[no-untyped-def]
            seen.update(checkpoint=checkpoint, device=device, **kwargs)
            return FakeTimesFM3()

    fake_module = type(sys)("timesfm3")
    fake_module.TimesFM3Forecaster = Loader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "timesfm3", fake_module)
    TimesFM3Backend("some/checkpoint", device="cuda", batch_size=64)
    assert seen == {
        "checkpoint": "some/checkpoint",
        "device": "cuda",
        "per_core_batch_size": 64,
    }


def test_timestral_backend_sends_pixels_in_chunks(tmp_path: Path) -> None:
    script = tmp_path / "bridge.py"
    script.write_text(FAKE_BRIDGE)
    backend = TimestralBackend(
        "any/model", "cpu", command=[sys.executable, str(script)], batch_size=2
    )
    try:
        context = np.arange(30, dtype="float32").reshape(5, 1, 6)
        q, _ = backend.forecast(context, None, None, horizon=3)
        assert backend._count == 3  # 2 + 2 + 1 pixels
        assert q.shape == (5, 1, 3, 3)
        np.testing.assert_allclose(q[:, 0, 0, 1], context[:, 0, -1] + 0.5)
    finally:
        backend.close()
