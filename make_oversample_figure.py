"""Reproduce the oversampling self-aliasing demonstration figure
(`outputs/figs/circe3_oversample_aliasing.png`): train a base-rate (1x) and a 2x
oversampled CIRCE3 on the BJT drive sweep and plot a hard-driven high-tone line
spectrum. The 1x model shows inharmonic alias spurs; 2x removes them.

Run: ``uv run python make_oversample_figure.py``
"""

from __future__ import annotations

import numpy as np
import torch

from vguitar import plotting as plot
from vguitar.circuits import get_circuit
from vguitar.config import Config, TrainConfig
from vguitar.data import Dataset
from vguitar.models.base import pick_device, to_inference_cpu
from vguitar.models.circe3 import CIRCE3
from vguitar.spice.runner import simulate


def _proc(m: CIRCE3, x: np.ndarray, c: np.ndarray) -> np.ndarray:
    pred = np.asarray(m.process(x, c), np.float32)
    lat = int(getattr(m, "latency_samples", 0) or 0)
    return pred[lat:] if lat else pred


def main() -> None:
    cfg = Config()
    sr = cfg.data.sr
    dev = pick_device()
    a, vb, _ = Dataset.load(cfg.paths.data / "bjt_bench_sweep.npz").split(0.12, 1e-4)
    circ = get_circuit("bjt")
    gq, f0 = 0.2, 2500.0

    def train(n_layers: int, oversample: int) -> CIRCE3:
        torch.manual_seed(0)
        m = CIRCE3(n_control=1, signal_idx=(0,), channels=24, n_blocks=2,
                   n_layers=n_layers, oversample=oversample, device=dev)
        m.fit(a, vb, TrainConfig(epochs=300, seq_len=4096, batch_size=12, lr=3e-3,
                                 warmup=2048, seed=0))
        to_inference_cpu(m)
        return m

    m1 = train(n_layers=8, oversample=1)
    m2 = train(n_layers=9, oversample=2)

    t = np.arange(int(0.2 * sr)) / sr
    tone = np.sin(2 * np.pi * f0 * t).astype(np.float32)
    yc = simulate(circ, (gq * tone).astype(np.float32), sr)
    preds = {
        "CIRCE3 1x — inharmonic alias spurs": _proc(m1, tone, np.array([gq], np.float32)),
        "CIRCE3 2x (oversampled) — spurs removed": _proc(m2, tone, np.array([gq], np.float32)),
    }
    fig = plot.fig_aliasing(yc, preds, sr, f0=f0, name=f"bjt @ {gq:g} V —")
    out = cfg.paths.outputs / "figs" / "circe3_oversample_aliasing.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
