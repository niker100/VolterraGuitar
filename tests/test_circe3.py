"""Contract + parity tests for CIRCE3 (the SOTA emulator).

CIRCE3 folds signal-acting controls into the input (so a pure-signal control is a
*gain*; the generic ``check_streaming`` harness, which passes no control, would
drive it with gain 0 — trivial). These tests therefore exercise the streaming
contract with explicit non-zero controls, across signal-only (unconditioned) and
signal+system (FiLM) control layouts, constant / moving / ragged block schedules,
plus the input-scaling identity, save/load, boundedness and contract flags.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vguitar.models.base import check_streaming_moving
from vguitar.models.circe3 import CIRCE3


def _const_stream_err(model: CIRCE3, c: np.ndarray, n: int = 4096, block: int = 128) -> float:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n).astype(np.float32) * 0.3
    y_off = model.process(x, c)
    model.reset()
    y_st = np.concatenate([model.process_block(x[i : i + block], c) for i in range(0, n, block)])[
        :n
    ]
    return float(np.max(np.abs(y_off - y_st)))


def _ragged_err(model: CIRCE3, c: np.ndarray, sizes=(37, 128, 200), n: int = 4096) -> float:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(n).astype(np.float32) * 0.3
    model.reset()
    out, pos, si = [], 0, 0
    while pos < n:
        b = sizes[si % len(sizes)]
        out.append(model.process_block(x[pos : pos + b], c))
        pos += b
        si += 1
    y_ragged = np.concatenate(out)[:n]
    model.reset()
    y_sample = np.concatenate([model.process_block(x[j : j + 1], c) for j in range(n)])[:n]
    return float(np.max(np.abs(y_ragged - y_sample)))


# (n_control, signal_idx) control layouts and a representative non-zero control.
CONFIGS = [
    (1, (0,)),  # signal-only, unconditioned (the BJT-drive / input-scaling case)
    (2, (0,)),  # drive (signal, folded) + tone (system, FiLM)
    (2, ()),  # both controls system (FiLM on both, no signal-fold)
]


@pytest.fixture(params=CONFIGS, ids=["signal_only", "signal+system", "system_only"])
def model_and_ctrl(request) -> tuple[CIRCE3, np.ndarray]:
    n_control, signal_idx = request.param
    torch.manual_seed(0)
    m = CIRCE3(n_control=n_control, signal_idx=signal_idx, channels=8, n_blocks=2, n_layers=4)
    c = np.full(n_control, 0.5, dtype=np.float32)
    return m, c


def test_streaming_constant_control(model_and_ctrl) -> None:
    m, c = model_and_ctrl
    err = _const_stream_err(m, c)
    assert err <= 1e-4, f"constant-control streaming err {err:.2e}"


def test_ragged_block(model_and_ctrl) -> None:
    m, c = model_and_ctrl
    err = _ragged_err(m, c)
    assert err <= 2e-3, f"ragged-block streaming err {err:.2e}"


def test_moving_control(model_and_ctrl) -> None:
    m, _ = model_and_ctrl
    err = check_streaming_moving(m, n=4096, block=128, atol=2e-3)
    assert err <= 2e-3, f"moving-control streaming err {err:.2e}"


def test_signal_only_is_input_scaling() -> None:
    """A pure-signal control must act EXACTLY as an input gain: process(x, [g])
    equals the unconditioned response to g*x — the whole point of input-scaling."""
    torch.manual_seed(0)
    m = CIRCE3(n_control=1, signal_idx=(0,), channels=8, n_blocks=2, n_layers=4)
    x = np.random.default_rng(2).standard_normal(2048).astype(np.float32) * 0.3
    y_g = m.process(x, np.array([0.7], np.float32))
    y_scaled = m.process(0.7 * x, np.array([1.0], np.float32))
    assert np.max(np.abs(y_g - y_scaled)) < 1e-5


def test_oversample_streaming_exact_and_latency() -> None:
    """2x internal oversampling must keep streaming bit-exact (process ==
    block-streamed) and report the polyphase group delay as latency_samples."""
    torch.manual_seed(0)
    m = CIRCE3(
        n_control=1, signal_idx=(0,), channels=8, n_blocks=2, n_layers=4, oversample=2, os_taps=127
    )
    assert m.latency_samples == (127 - 1) // 2
    c = np.array([0.6], np.float32)
    err = _const_stream_err(m, c)
    assert err <= 1e-4, f"oversampled streaming err {err:.2e}"
    # output length equals input length (block-size preserved through up/down)
    x = np.random.default_rng(0).standard_normal(2048).astype(np.float32) * 0.3
    assert len(m.process(x, c)) == len(x)


def test_oversample_save_load_roundtrip(tmp_path) -> None:
    """Oversampling hparams round-trip and reproduce the streamed output."""
    torch.manual_seed(0)
    m = CIRCE3(n_control=1, signal_idx=(0,), channels=8, n_blocks=2, n_layers=4, oversample=2)
    p = tmp_path / "os.model"
    m.save(p)
    r = CIRCE3.load(p)
    assert r.oversample == 2 and r.os_taps == m.os_taps
    x = np.random.default_rng(1).standard_normal(2048).astype(np.float32) * 0.3
    c = np.array([0.5], np.float32)
    assert np.max(np.abs(m.process(x, c) - r.process(x, c))) < 1e-6


def test_rect_input_features_streaming_and_roundtrip(tmp_path) -> None:
    """Rectified input features (the discontinuity-basis fix) must keep streaming
    bit-exact, preserve output length, and round-trip through save/load."""
    torch.manual_seed(0)
    m = CIRCE3(
        n_control=1, signal_idx=(0,), channels=8, n_blocks=2, n_layers=4, rect_thr=(0.0, 0.5, 1.0)
    )
    c = np.array([0.6], np.float32)
    assert _const_stream_err(m, c) <= 1e-4
    x = np.random.default_rng(0).standard_normal(2048).astype(np.float32) * 0.3
    assert len(m.process(x, c)) == len(x)
    p = tmp_path / "rect.model"
    m.save(p)
    r = CIRCE3.load(p)
    assert r.rect_thr == m.rect_thr
    assert np.max(np.abs(m.process(x, c) - r.process(x, c))) < 1e-6


def test_rect_input_preserves_input_scaling() -> None:
    """Rectified features are positive-homogeneous on the scaled input, so the
    signal-control = input-gain identity still holds exactly."""
    torch.manual_seed(0)
    m = CIRCE3(
        n_control=1, signal_idx=(0,), channels=8, n_blocks=2, n_layers=4, rect_thr=(0.0, 0.5)
    )
    x = np.random.default_rng(2).standard_normal(2048).astype(np.float32) * 0.3
    y_g = m.process(x, np.array([0.7], np.float32))
    y_scaled = m.process(0.7 * x, np.array([1.0], np.float32))
    assert np.max(np.abs(y_g - y_scaled)) < 1e-5


def test_iir_state_streaming_and_roundtrip(tmp_path) -> None:
    """Leaky-integrator state channels (n_state>0) keep streaming bit-exact through
    the full oversampling chain, preserve output length, and round-trip save/load."""
    torch.manual_seed(0)
    m = CIRCE3(n_control=1, signal_idx=(0,), channels=8, n_blocks=2, n_layers=4,
               oversample=2, n_state=4)
    c = np.array([0.6], np.float32)
    assert _const_stream_err(m, c) <= 1e-4
    x = np.random.default_rng(0).standard_normal(2048).astype(np.float32) * 0.3
    assert len(m.process(x, c)) == len(x)
    p = tmp_path / "iir.model"
    m.save(p)
    r = CIRCE3.load(p)
    assert r.n_state == 4
    assert np.max(np.abs(m.process(x, c) - r.process(x, c))) < 1e-6


def test_iir_state_ragged_blocks() -> None:
    """The one-pole state carries correctly across ragged block sizes (block-size
    invariant), matching per-sample streaming."""
    torch.manual_seed(0)
    m = CIRCE3(n_control=1, signal_idx=(0,), channels=8, n_blocks=2, n_layers=4, n_state=4)
    assert _ragged_err(m, np.array([0.6], np.float32)) <= 1e-4


def test_iir_state_preserves_input_scaling() -> None:
    """The one-pole state is linear in the gain-scaled input, so the signal-control =
    input-gain identity still holds."""
    torch.manual_seed(0)
    m = CIRCE3(n_control=1, signal_idx=(0,), channels=8, n_blocks=2, n_layers=4, n_state=4)
    x = np.random.default_rng(2).standard_normal(2048).astype(np.float32) * 0.3
    y_g = m.process(x, np.array([0.7], np.float32))
    y_scaled = m.process(0.7 * x, np.array([1.0], np.float32))
    assert np.max(np.abs(y_g - y_scaled)) < 1e-5


def test_iir_state_tau_range(tmp_path) -> None:
    """state_tau_s widens the one-pole tau init range, survives save/load, and keeps
    streaming exact; old checkpoints without the key fall back to the default."""
    torch.manual_seed(0)
    m = CIRCE3(n_control=1, signal_idx=(0,), channels=8, n_blocks=2, n_layers=4,
               n_state=4, state_tau_s=(5e-3, 2.0))
    torch.manual_seed(0)
    ref = CIRCE3(n_control=1, signal_idx=(0,), channels=8, n_blocks=2, n_layers=4,
                 n_state=4)
    # the slowest pole must sit closer to 1 than the default 500 ms one
    assert float(torch.sigmoid(m.net.a_logit[-1])) > float(torch.sigmoid(ref.net.a_logit[-1]))
    assert _const_stream_err(m, np.array([0.6], np.float32)) <= 1e-4
    p = tmp_path / "tau.model"
    m.save(p)
    r = CIRCE3.load(p)
    assert r.state_tau_s == (5e-3, 2.0)
    x = np.random.default_rng(1).standard_normal(2048).astype(np.float32) * 0.3
    c = np.array([0.6], np.float32)
    assert np.max(np.abs(m.process(x, c) - r.process(x, c))) < 1e-6


def test_varpro_training_streaming_exact() -> None:
    """VarPro training (closed-form readout solved by lstsq each step) produces a model
    that still streams bit-exact — out[3] stays a plain linear conv, set by the global
    solve — and runs with the IIR memory channels (the memory-heavy-circuit path)."""
    from vguitar.config import TrainConfig
    from vguitar.data import Dataset

    rng = np.random.default_rng(0)
    n = 8000
    x = (rng.standard_normal(n) * 0.3).astype(np.float32)
    y = np.tanh(3.0 * x).astype(np.float32)
    ds = Dataset(x, y, 44_100, controls=np.ones((n, 1), np.float32))
    torch.manual_seed(0)
    m = CIRCE3(n_control=1, signal_idx=(0,), channels=8, n_blocks=1, n_layers=4,
               n_state=4, dcblock_fc=0.0)
    m.fit(ds, ds, TrainConfig(epochs=3, lr=3e-3, seq_len=2048, batch_size=8, warmup=512,
                              varpro=True))
    assert _const_stream_err(m, np.array([1.0], np.float32)) <= 1e-4


def test_block_act_mixed_streaming_and_roundtrip(tmp_path) -> None:
    """The heterogeneous mixed-activation block (block_act='mixed') must keep
    streaming bit-exact (incl. under oversampling + FiLM), preserve output length,
    and round-trip through save/load."""
    torch.manual_seed(0)
    # signal + system control so the FiLM path is exercised alongside the mixed block
    m = CIRCE3(
        n_control=2,
        signal_idx=(0,),
        channels=15,
        n_blocks=2,
        n_layers=4,
        oversample=2,
        block_act="mixed",
    )
    c = np.array([0.6, 0.3], np.float32)
    assert _const_stream_err(m, c) <= 1e-4
    x = np.random.default_rng(0).standard_normal(2048).astype(np.float32) * 0.3
    assert len(m.process(x, c)) == len(x)
    p = tmp_path / "mixed.model"
    m.save(p)
    r = CIRCE3.load(p)
    assert r.block_act == "mixed"
    assert np.max(np.abs(m.process(x, c) - r.process(x, c))) < 1e-6


def test_block_act_mixed_preserves_input_scaling() -> None:
    """The mixed-activation groups act per-channel after the conv, so the
    signal-control = input-gain identity still holds exactly."""
    torch.manual_seed(0)
    m = CIRCE3(n_control=1, signal_idx=(0,), channels=15, n_blocks=2, n_layers=4, block_act="mixed")
    x = np.random.default_rng(2).standard_normal(2048).astype(np.float32) * 0.3
    y_g = m.process(x, np.array([0.7], np.float32))
    y_scaled = m.process(0.7 * x, np.array([1.0], np.float32))
    assert np.max(np.abs(y_g - y_scaled)) < 1e-5


def test_block_act_invalid() -> None:
    with pytest.raises(ValueError, match="block_act"):
        CIRCE3(n_control=1, channels=8, n_blocks=1, n_layers=2, block_act="bogus")


def test_out_shaper_fourier_streaming_and_roundtrip(tmp_path) -> None:
    """The learned Fourier waveshaper head (out_shaper='fourier') must keep streaming
    bit-exact (incl. under oversampling), preserve length, and round-trip save/load.
    Random (non-zero) shaper weights are set so the head is actually exercised."""
    torch.manual_seed(0)
    m = CIRCE3(
        n_control=1,
        signal_idx=(0,),
        channels=12,
        n_blocks=2,
        n_layers=4,
        oversample=2,
        out_shaper="fourier",
        shaper_k=6,
    )
    # zero-init c_k => identity; perturb so the periodic head is non-trivial
    with torch.no_grad():
        m.net.shaper_c.copy_(torch.linspace(0.1, -0.1, 6))
        m.net.shaper_w.copy_(torch.tensor([1.3]))
    c = np.array([0.6], np.float32)
    assert _const_stream_err(m, c) <= 1e-4
    x = np.random.default_rng(0).standard_normal(2048).astype(np.float32) * 0.3
    assert len(m.process(x, c)) == len(x)
    p = tmp_path / "shaper.model"
    m.save(p)
    r = CIRCE3.load(p)
    assert r.out_shaper == "fourier" and r.shaper_k == 6
    assert np.max(np.abs(m.process(x, c) - r.process(x, c))) < 1e-6


def test_out_shaper_preserves_input_scaling() -> None:
    """The Fourier head is a pointwise function of the net output, so the
    signal-control = input-gain identity still holds exactly."""
    torch.manual_seed(0)
    m = CIRCE3(
        n_control=1,
        signal_idx=(0,),
        channels=12,
        n_blocks=2,
        n_layers=4,
        out_shaper="fourier",
        shaper_k=6,
    )
    with torch.no_grad():
        m.net.shaper_c.copy_(torch.linspace(0.1, -0.1, 6))
        m.net.shaper_w.copy_(torch.tensor([1.3]))
    x = np.random.default_rng(2).standard_normal(2048).astype(np.float32) * 0.3
    y_g = m.process(x, np.array([0.7], np.float32))
    y_scaled = m.process(0.7 * x, np.array([1.0], np.float32))
    assert np.max(np.abs(y_g - y_scaled)) < 1e-5


def test_out_shaper_invalid() -> None:
    with pytest.raises(ValueError, match="out_shaper"):
        CIRCE3(n_control=1, channels=8, n_blocks=1, n_layers=2, out_shaper="bogus")


def test_fit_smoke_runs() -> None:
    """A short fit() must run end-to-end. The no-ngspice suite otherwise never
    exercises the training loop (only streaming/save-load on untrained nets), so a
    fit() regression — e.g. a stray reference to a removed option — would slip
    through. Guards the grad-clip path + post-fit streaming too."""
    from vguitar.config import TrainConfig
    from vguitar.data import Dataset

    rng = np.random.default_rng(0)
    n = 12000
    x = (rng.standard_normal(n) * 0.3).astype(np.float32)
    y = np.tanh(2.0 * x).astype(np.float32)
    c = np.full((n, 1), 0.5, np.float32)
    ds = Dataset(x, y, 44100, controls=c)
    m = CIRCE3(n_control=1, signal_idx=(0,), channels=8, n_blocks=1, n_layers=4)
    rep = m.fit(ds, ds, TrainConfig(epochs=2, seq_len=4096, batch_size=8, warmup=1024))
    assert np.isfinite(rep.info["best_val_esr"])
    y_out = m.process(x[:2048], np.array([0.5], np.float32))
    assert len(y_out) == 2048
    assert np.all(np.isfinite(y_out))


def test_grad_clip_default_and_roundtrip(tmp_path) -> None:
    """grad_clip defaults on (1.0) and round-trips through save/load (it is a
    training-only hyperparameter, so inference is unaffected by its value)."""
    m = CIRCE3(n_control=1, channels=8, n_blocks=1, n_layers=2)
    assert m.grad_clip == 1.0
    m2 = CIRCE3(n_control=1, channels=8, n_blocks=1, n_layers=2, grad_clip=0.0)
    p = tmp_path / "gc.model"
    m2.save(p)
    assert CIRCE3.load(p).grad_clip == 0.0


def test_save_load_roundtrip(model_and_ctrl, tmp_path) -> None:
    m, c = model_and_ctrl
    p = tmp_path / "circe3.model"
    m.save(p)
    r = CIRCE3.load(p)
    assert r.signal_idx == m.signal_idx
    x = np.random.default_rng(3).standard_normal(2048).astype(np.float32) * 0.3
    assert np.max(np.abs(m.process(x, c) - r.process(x, c))) < 1e-6


def test_hot_input_bounded(model_and_ctrl) -> None:
    m, c = model_and_ctrl
    m.net.out_bound.copy_(torch.tensor(0.5))
    m.reset()
    x = (50.0 * np.random.default_rng(4).standard_normal(2048)).astype(np.float32)
    y = m.process(x, c)
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) <= 0.5 + 1e-4


def test_contract_flags() -> None:
    m = CIRCE3(n_control=1)
    assert m.conditioned is True
    assert m.latency_samples == 0
    assert m.num_params() > 0
    assert CIRCE3(n_control=1, saturator="adaa2").latency_samples == 1


def test_invalid_saturator() -> None:
    with pytest.raises(ValueError, match="saturator"):
        CIRCE3(n_control=1, saturator="bogus")
