"""Campaign 4 prototype (rank-3 lever) — do LEARNABLE-threshold rectified corners
crack the sharp-knee circuits (hard_clipper) and the dead-zone (crossover)?

A finite-Lipschitz smooth-activation TCN cannot synthesize a true slope discontinuity
(hard-clip knee, dead-zone edge) — it rounds it (spectral bias). Exact rectified
features relu(x - t), relu(-x - t), abs(x) ARE discontinuities the net can combine
linearly, leaving it to learn smooth dynamics around the corner. Campaign 1 found a
FIXED bank (0.1,0.3,0.6,1.0) inert — because the knees live at circuit-specific volts
and the fixed grid missed them (hard_clipper's input is ±0.3 V, so 0.6/1.0 never fire).

This makes the thresholds LEARNABLE (t_k = softplus(param), K=4, init 0.05..1.5 V), so
SGD places the corners at each circuit's actual knee. Pointwise ⇒ streaming-exact and
input-scaling-consistent (fixed-volt knee, as the physics demands). Uniform: the same K
learnable thresholds for every circuit; circuits with no knee learn the corner weights
toward zero in the input conv. A/B: corners ON vs OFF at OS1 on hard_clipper + crossover
+ bjt guard. If it helps, integrate into circe3.py (replacing the dead fixed rect_thr)
with the numpy twin.

Run (background): uv run python -m experiments.sota.corner_probe
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.common import held_esr, make_log
from vguitar.config import TrainConfig
from vguitar.data import Dataset
from vguitar.models.circe3 import CIRCE3
from vguitar.models.tcn import _MixedLayer

EPOCHS = 150
K_CORNERS = 4
T_INIT = (0.05, 0.3, 0.7, 1.5)  # initial corner thresholds (volts), then learned
CIRCUITS = [
    ("hard_clipper", "hard_clipper_edge_sweep", "hard_clipper_edge_test"),
    ("crossover", "crossover_classb_edge_sweep", "crossover_classb_edge_test"),
    ("bjt", "bjt_bench_sweep", "bjt_bench_fp_test"),
]


def _corner_feats(x: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
    """``x`` (B, T), learnable ``thr`` (K,) -> (B, 2K+2, T) feature stack:
    [x, relu(x-t_k), relu(-x-t_k) for k, abs(x)]. Symmetric corner pair at +/-t_k."""
    t = thr[None, :, None]  # (1, K, 1)
    xe = x[:, None, :]  # (B, 1, T)
    pos = F.relu(xe - t)  # (B, K, T) positive knee
    neg = F.relu(-xe - t)  # (B, K, T) negative knee
    return torch.cat([x[:, None, :], pos, neg, x.abs()[:, None, :]], dim=1)


class _CornerNet(nn.Module):
    """Mixed-activation TCN with optional learnable-threshold rectified corner inputs."""

    out_bound: torch.Tensor

    def __init__(self, channels: int, n_layers: int, k_corners: int = K_CORNERS,
                 use_corners: bool = True, kernel: int = 3) -> None:
        super().__init__()
        self.use_corners = use_corners
        self.k = k_corners if use_corners else 0
        n_in = (2 * self.k + 2) if use_corners else 1
        self.input = nn.Conv1d(n_in, channels, 1)
        self.layers = nn.ModuleList(_MixedLayer(channels, kernel, 2**i) for i in range(n_layers))
        self.head = nn.Sequential(
            nn.ReLU(), nn.Conv1d(channels, channels, 1), nn.ReLU(), nn.Conv1d(channels, 1, 1)
        )
        if use_corners:
            t0 = torch.tensor(T_INIT[:k_corners], dtype=torch.float32)
            # softplus^-1 so softplus(raw)=t0 -> thresholds stay positive when learned
            self.thr_raw = nn.Parameter(torch.log(torch.expm1(t0)))
        self.register_buffer("out_bound", torch.tensor(1.0))

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        feats = _corner_feats(x, F.softplus(self.thr_raw)) if self.use_corners else x.unsqueeze(1)
        h = self.input(feats)
        skip = torch.zeros_like(h)
        for layer in self.layers:
            h, sk = layer(h)
            skip = skip + sk
        return self.head(skip).squeeze(1)

    def forward(self, x: torch.Tensor, c_sys: torch.Tensor | None = None) -> torch.Tensor:
        a = float(self.out_bound)
        return torch.clamp(self.raw(x), -a, a)


class CornerState(CIRCE3):
    """CIRCE3 with learnable rectified corners; OS1 torch-offline process (prototype)."""

    name = "corner_state"

    def __init__(self, channels: int = 24, n_layers: int = 9, k_corners: int = K_CORNERS,
                 use_corners: bool = True, device: str = "cpu", **kw: Any) -> None:
        super().__init__(n_control=1, signal_idx=(0,), channels=channels, n_blocks=1,
                         n_layers=1, oversample=1, dcblock_fc=0.0, device=device, **kw)
        self.net = _CornerNet(channels, n_layers, k_corners, use_corners).to(self.device)

    def process(self, x: np.ndarray, c: np.ndarray | None = None) -> np.ndarray:
        self.net.eval()
        cv = self._control_vec(c)
        g = self._signal_gain(cv)
        xb = (np.ascontiguousarray(x, np.float32).reshape(-1) * g).astype(np.float32)
        with torch.no_grad():
            o = self.net(torch.from_numpy(xb).to(self.device).view(1, -1)).view(-1).cpu().numpy()
        a = float(self.net.out_bound)
        return np.clip(o, -a, a).astype(np.float32)


def _selftest() -> None:
    """Features have the right shape and gradient flows to the thresholds."""
    net = _CornerNet(8, 2, use_corners=True)
    x = torch.randn(2, 64)
    y = net(x)
    assert y.shape == (2, 64)
    net.zero_grad()
    y.abs().sum().backward()
    assert net.thr_raw.grad is not None and torch.isfinite(net.thr_raw.grad).all()
    print(f"corner self-test OK (thr init {F.softplus(net.thr_raw).tolist()})", flush=True)


def main() -> None:
    log = make_log("sota_corner_probe")
    _selftest()
    log(f"learnable-corner A/B: K={K_CORNERS} t_init={T_INIT}, {EPOCHS}ep OS1, ch24/L9")
    results: dict[str, Any] = {}
    out = Path("outputs/sota/corner_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    for key, sweep_nm, test_nm in CIRCUITS:
        tr = Dataset.load(f"data/{sweep_nm}.npz")
        ts = Dataset.load(f"data/{test_nm}.npz")
        row: dict[str, Any] = {"kind": "smooth" if key == "bjt" else "hard"}
        for tag, use in (("no_corner", False), ("corners", True)):
            t = time.time()
            torch.manual_seed(0)
            m = CornerState(channels=24, n_layers=9, use_corners=use, device="cuda")
            m.fit(tr, ts, TrainConfig(epochs=EPOCHS, lr=3e-3, seq_len=4096,
                                      batch_size=12, warmup=2048, seed=0))
            esr = held_esr(m, ts)
            thr = (F.softplus(m.net.thr_raw).tolist() if use else None)
            row[tag] = {"held": esr, "params": int(m.num_params()), "thr": thr,
                        "secs": time.time() - t}
            log(f"{key:14s} {tag:9s} held={esr:.4f} thr={thr} ({time.time()-t:.0f}s)")
        d = row["no_corner"]["held"]
        imp = 100.0 * (row["corners"]["held"] - d) / d if d else 0.0
        log(f"  -> {key}: {d:.4f} -> {row['corners']['held']:.4f} ({imp:+.0f}%)")
        results[key] = row
        out.write_text(json.dumps(results, indent=2))
    log("wrote outputs/sota/corner_probe.json")


if __name__ == "__main__":
    main()
