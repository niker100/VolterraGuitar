"""Contract tests for every registered model.

For each model: a minimal fit on a short, learnable synthetic dataset, then the
two invariants every model must satisfy — blockwise streaming equals offline
processing (``check_streaming``), and ``save``/``load`` round-trips exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import lfilter

from vguitar.config import TrainConfig
from vguitar.data import Dataset
from vguitar.models import all_models, get_model
from vguitar.models.base import check_streaming

SR = 8000
# Conditioned models (CIRCE) need an exogenous-control dataset; they have their
# own test (test_circe.py). The generic contract test covers unconditioned ones.
MODELS = sorted(n for n, cls in all_models().items() if not cls.conditioned)


@pytest.fixture(scope="module")
def toy() -> Dataset:
    """A short nonlinear-with-memory target: tanh saturation after a one-pole LPF."""
    rng = np.random.default_rng(0)
    x = (0.5 * rng.standard_normal(8192)).astype(np.float32)
    lp = lfilter([0.2], [1.0, -0.8], x)
    y = np.tanh(2.0 * lp).astype(np.float32)
    return Dataset(x, y, SR, name="toy")


@pytest.mark.parametrize("name", MODELS)
def test_model_fit_stream_roundtrip(name: str, toy: Dataset, tmp_path) -> None:
    import torch

    torch.manual_seed(0)
    cfg = TrainConfig(seq_len=1024, warmup=128, batch_size=8, epochs=1, sr=SR)
    train, val, _ = toy.split(0.2, 0.2)

    model = get_model(name)()
    model.fit(train, val, cfg)

    # Invariant 1: streaming == offline (within a small numerical tolerance).
    check_streaming(model, n=2048, block=128, atol=2e-3)

    # Invariant 2: save/load preserves the processed output exactly.
    path = tmp_path / f"{name}.model"
    model.save(path)
    reloaded = get_model(name).load(path)
    x = toy.x[:2048]
    a = np.asarray(model.process(x), dtype=np.float32)
    b = np.asarray(reloaded.process(x), dtype=np.float32)
    assert np.max(np.abs(a - b)) < 1e-4
