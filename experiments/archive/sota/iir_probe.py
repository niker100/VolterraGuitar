"""Campaign 3 prototype (rank-2 lever) — does a leaky-integrator IIR state channel
fix the structural memory deficit on hysteretic_fuzz / crossover?

CIRCE3's dilated-FIR receptive field is ~21 ms (n_layers=9, OS2), but hysteretic_fuzz's
bias-recovery time constant is Rb*Cb = 22 ms (bias=0) to 440 ms (bias=1). The FIR stack
physically cannot see the slow state that defines the hysteresis loop — no capacity /
loss / oversampling lever adds a missing pole; only added IIR/integrator state can.

This prepends K learnable one-pole state channels to the TCN input:
    s_k[n] = a_k * s_k[n-1] + (1 - a_k) * x[n]
with a_k = sigmoid(logit_k), time constants initialised to span ~5..500 ms. The one-pole
is a first-order LINEAR recurrence, so it is computed with a stable log-depth parallel
scan (~12 vectorised steps for T=4096) — fully differentiable, GPU-efficient, and
trivially streamable in a numpy twin on integration. It is O(1)/sample (real-time-cheap),
uniform (smooth circuits learn a~0), and input-scaling-equivariant (the state is linear
in the gain-scaled input). A/B: state ON vs OFF at OS1 on hysteretic_fuzz + crossover
(memory/dead-zone) + bjt (smooth regression guard). If it helps, integrate into
circe3.py with the numpy streaming twin + a streaming-exactness test.

Run (background): uv run python -m experiments.sota.iir_probe
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from experiments.common import held_esr, make_log
from vguitar.config import TrainConfig
from vguitar.data import Dataset
from vguitar.models.archive.circe3 import CIRCE3
from vguitar.models.archive.tcn import _MixedLayer

EPOCHS = 150
SR = 44_100
CIRCUITS = [
    ("hysteretic_fuzz", "hysteretic_fuzz_edge_sweep", "hysteretic_fuzz_edge_test"),
    ("crossover", "crossover_classb_edge_sweep", "crossover_classb_edge_test"),
    ("bjt", "bjt_bench_sweep", "bjt_bench_fp_test"),
]
TAUS_MS = (5.0, 25.0, 100.0, 500.0)  # one-pole time constants the state bank spans


def _onepole_scan(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Stable log-depth parallel scan of s_k[n] = a_k s_k[n-1] + (1-a_k) x[n].

    ``x`` (B, T), ``a`` (K,) in (0,1) -> ``s`` (B, K, T). Hillis-Steele inclusive
    scan: after the step at offset d, s[n] = sum_{j<=n} a^j (1-a) x[n-j] up to 2d
    terms; doubling d each step reaches the full window in ceil(log2 T) steps. Each
    step reads the pre-step tensor and writes a fresh one (no in-place aliasing)."""
    t = x.shape[1]
    s = (1.0 - a)[None, :, None] * x[:, None, :]  # (B, K, T) input term
    shift, a_pow = 1, a.clone()  # a_pow tracks a^shift
    while shift < t:
        tail = s[..., shift:] + a_pow[None, :, None] * s[..., :-shift]
        s = torch.cat([s[..., :shift], tail], dim=-1)
        shift *= 2
        a_pow = a_pow * a_pow
    return s


class _IIRStateNet(nn.Module):
    """Mixed-activation TCN with K optional learnable one-pole state input channels."""

    out_bound: torch.Tensor

    def __init__(self, channels: int, n_layers: int, k_state: int = 4,
                 use_state: bool = True, kernel: int = 3, sr: int = SR) -> None:
        super().__init__()
        self.use_state = use_state
        self.k_state = k_state if use_state else 0
        self.input = nn.Conv1d(1 + self.k_state, channels, 1)
        self.layers = nn.ModuleList(_MixedLayer(channels, kernel, 2**i) for i in range(n_layers))
        self.head = nn.Sequential(
            nn.ReLU(), nn.Conv1d(channels, channels, 1), nn.ReLU(), nn.Conv1d(channels, 1, 1)
        )
        if use_state:
            taus = torch.tensor(TAUS_MS[:k_state], dtype=torch.float32) * 1e-3
            a0 = torch.exp(-1.0 / (taus * sr)).clamp(1e-4, 1 - 1e-6)
            self.a_logit = nn.Parameter(torch.log(a0 / (1.0 - a0)))  # learnable decays
        self.register_buffer("out_bound", torch.tensor(1.0))

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        feats = x.unsqueeze(1)
        if self.use_state:
            feats = torch.cat([feats, _onepole_scan(x, torch.sigmoid(self.a_logit))], dim=1)
        h = self.input(feats)
        skip = torch.zeros_like(h)
        for layer in self.layers:
            h, sk = layer(h)
            skip = skip + sk
        return self.head(skip).squeeze(1)

    def forward(self, x: torch.Tensor, c_sys: torch.Tensor | None = None) -> torch.Tensor:
        a = float(self.out_bound)
        return torch.clamp(self.raw(x), -a, a)


class IIRState(CIRCE3):
    """CIRCE3 with the IIR state bank; OS1 torch-offline process (prototype)."""

    name = "iir_state"

    def __init__(self, channels: int = 24, n_layers: int = 9, k_state: int = 4,
                 use_state: bool = True, device: str = "cpu", **kw: Any) -> None:
        super().__init__(n_control=1, signal_idx=(0,), channels=channels, n_blocks=1,
                         n_layers=1, oversample=1, dcblock_fc=0.0, device=device, **kw)
        self.net = _IIRStateNet(channels, n_layers, k_state, use_state).to(self.device)

    def process(self, x: np.ndarray, c: np.ndarray | None = None) -> np.ndarray:
        self.net.eval()
        cv = self._control_vec(c)
        g = self._signal_gain(cv)
        xb = (np.ascontiguousarray(x, np.float32).reshape(-1) * g).astype(np.float32)
        with torch.no_grad():
            o = self.net(torch.from_numpy(xb).to(self.device).view(1, -1)).view(-1).cpu().numpy()
        a = float(self.net.out_bound)
        return np.clip(o, -a, a).astype(np.float32)


def _selftest_scan() -> None:
    """Assert the parallel scan matches a sequential one-pole reference."""
    torch.manual_seed(0)
    x = torch.randn(3, 257)
    a = torch.tensor([0.5, 0.9, 0.999])
    par = _onepole_scan(x, a)
    ref = torch.zeros_like(par)
    for n in range(x.shape[1]):
        prev = ref[:, :, n - 1] if n else 0.0
        ref[:, :, n] = a[None, :] * prev + (1.0 - a)[None, :] * x[:, None, n]
    err = (par - ref).abs().max().item()
    assert err < 1e-4, f"scan mismatch {err}"
    print(f"scan self-test OK (max err {err:.2e})", flush=True)


def main() -> None:
    log = make_log("sota_iir_probe")
    _selftest_scan()
    log(f"IIR state A/B: k_state={len(TAUS_MS)} taus(ms)={TAUS_MS}, {EPOCHS}ep OS1, ch24/L9")
    results: dict[str, Any] = {}
    out = Path("outputs/sota/iir_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    for key, sweep_nm, test_nm in CIRCUITS:
        tr = Dataset.load(f"data/{sweep_nm}.npz")
        ts = Dataset.load(f"data/{test_nm}.npz")
        row: dict[str, Any] = {"kind": "smooth" if key == "bjt" else "hard"}
        for tag, use_state in (("no_state", False), ("iir_state", True)):
            t = time.time()
            torch.manual_seed(0)
            m = IIRState(channels=24, n_layers=9, use_state=use_state, device="cuda")
            m.fit(tr, ts, TrainConfig(epochs=EPOCHS, lr=3e-3, seq_len=4096,
                                      batch_size=12, warmup=2048, seed=0))
            esr = held_esr(m, ts)
            row[tag] = {"held": esr, "params": int(m.num_params()), "secs": time.time() - t}
            log(f"{key:16s} {tag:9s} held={esr:.4f} params={m.num_params()} "
                f"({time.time()-t:.0f}s)")
        d = row["no_state"]["held"]
        imp = 100.0 * (row["iir_state"]["held"] - d) / d if d else 0.0
        log(f"  -> {key}: {d:.4f} -> {row['iir_state']['held']:.4f} ({imp:+.0f}%)")
        results[key] = row
        out.write_text(json.dumps(results, indent=2))
    log("wrote outputs/sota/iir_probe.json")


if __name__ == "__main__":
    main()
