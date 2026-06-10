"""CIRCE-X — the multi-modal / mixture-of-experts swing (frontier-plan Step 2).

Hypothesis: no SINGLE inductive bias is best for every circuit (smooth-saturating
wants tanh smoothness; sharp wants corners; folds want periodicity; memory-circuits
want long dilation; near-static wants shallow). So fuse heterogeneous experts in
PARALLEL and let a learned per-sample gate pick the mix — one architecture,
trained circuit-agnostically, that *contains* every approach. The headline is
broad cross-circuit generalization: MEAN and WORST-CASE held-ESR over the suite,
NOT any single circuit.

This is a CONTAINED PROTOTYPE to validate the hypothesis before any streaming-exact
integration:
  * `_CirceXNet`: shared input projection -> N parallel branches (each a small TCN
    with its own activation family / depth / memory) -> per-branch scalar head ->
    a learned per-sample softmax GATE over branches -> mixed output.
  * `CIRCEX(CIRCE3)`: subclasses CIRCE3 so it REUSES CIRCE3.fit verbatim (same
    windows, input-scaling, pre-emph+ESR loss, cosine-LR, grad-clip) — only the net
    and an offline `process` differ. Run at OS1 for BOTH models here (isolates the
    MoE question from oversampling; add OS2 to both later if MoE wins).

A/B vs the current default (single-branch mixed TCN), same harness, across the
suite (smooth + folds + dead-zone). Reports per-circuit held-ESR + mean + worst +
params + the learned gate weights (which expert each circuit leans on). RTF is
deferred to integration (the prototype is offline-torch); the kitchen-sink will be
big — if it generalizes, the next step is ablate->prune to a real-time subset.

Run: uv run python circex_probe.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from vguitar import metrics as M
from vguitar.config import TrainConfig
from vguitar.data import Dataset
from vguitar.models.archive.circe3 import CIRCE3, _segments
from vguitar.models.archive.tcn import _GatedLayer, _MixedLayer

# The heterogeneous expert branches: (activation kind, n_layers) — long-smooth,
# corner-capable, and shallow-short. Circuit-agnostic; the gate learns the mix.
BRANCHES = [("gated", 9), ("mixed", 7), ("gated", 4)]
EPOCHS = 150

CIRCUITS = [
    ("bjt", "bjt_bench_sweep", "bjt_bench_fp_test", "smooth", 1),
    ("jfet", "jfet_bench_sweep", "jfet_bench_fp_test", "smooth", 1),
    ("tube_screamer", "tube_screamer_bench_sweep", "tube_screamer_bench_fp_test", "smooth", 1),
    ("crossover", "crossover_classb_edge_sweep", "crossover_classb_edge_test", "hard", 2),
    ("wavefolder", "wavefolder_edge_sweep", "wavefolder_edge_test", "hard", 2),
]
SEEDS = (0, 7)

LOG = Path("outputs/logs/circex_probe.log")
LOG.parent.mkdir(parents=True, exist_ok=True)
_t0 = time.time()


def log(msg: str) -> None:
    line = f"[{time.time() - _t0:7.1f}s] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


class _CirceXNet(nn.Module):
    """Shared input -> N heterogeneous branch-TCNs -> per-sample softmax gate -> mix."""

    out_bound: torch.Tensor

    def __init__(self, channels: int, branches: list[tuple[str, int]], kernel: int = 3) -> None:
        super().__init__()
        self.input = nn.Conv1d(1, channels, 1)
        self.branches = nn.ModuleList()
        self.heads = nn.ModuleList()
        for kind, nl in branches:
            cls = _MixedLayer if kind == "mixed" else _GatedLayer
            self.branches.append(nn.ModuleList(cls(channels, kernel, 2**i) for i in range(nl)))
            self.heads.append(
                nn.Sequential(
                    nn.ReLU(), nn.Conv1d(channels, channels, 1),
                    nn.ReLU(), nn.Conv1d(channels, 1, 1),
                )
            )
        # Per-sample gate: softmax over branches from the shared input features.
        self.gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1), nn.ReLU(), nn.Conv1d(channels, len(branches), 1)
        )
        self.register_buffer("out_bound", torch.tensor(1.0))

    def raw(self, x: torch.Tensor, c_sys: torch.Tensor | None = None) -> torch.Tensor:
        h0 = self.input(x.unsqueeze(1))  # (B, C, T)
        preds = []
        for layers, head in zip(self.branches, self.heads, strict=True):
            h = h0
            skip = torch.zeros_like(h)
            for layer in layers:
                h, sk = layer(h)
                skip = skip + sk
            preds.append(head(skip))  # (B, 1, T)
        pred = torch.cat(preds, dim=1)  # (B, n_branch, T)
        w = torch.softmax(self.gate(h0), dim=1)  # (B, n_branch, T)
        return (w * pred).sum(1)  # (B, T)

    def gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Mean per-branch gate weight over time (for interpretability)."""
        h0 = self.input(x.unsqueeze(1))
        return torch.softmax(self.gate(h0), dim=1).mean(dim=(0, 2))

    def forward(self, x: torch.Tensor, c_sys: torch.Tensor | None = None) -> torch.Tensor:
        a = float(self.out_bound)
        return torch.clamp(self.raw(x, c_sys), -a, a)


class CIRCEX(CIRCE3):
    """Multi-branch MoE over CIRCE3's machinery. Reuses CIRCE3.fit; OS1 offline eval."""

    name = "circex"

    def __init__(self, branches: list[tuple[str, int]], channels: int = 24,
                 n_control: int = 1, signal_idx: tuple[int, ...] = (0,),
                 device: str = "cpu", **kw: Any) -> None:
        # Build CIRCE3 (tiny placeholder spine) to set up every attr fit() needs,
        # then swap in the MoE net. oversample=1 (prototype isolates the MoE question).
        super().__init__(n_control=n_control, signal_idx=signal_idx, channels=channels,
                         n_blocks=1, n_layers=1, oversample=1, device=device, **kw)
        self.branches_spec = branches
        self.net = _CirceXNet(channels, branches).to(self.device)

    def process(self, x: np.ndarray, c: np.ndarray | None = None) -> np.ndarray:
        """Offline torch forward (OS1) + 1 Hz DC-block, matching CIRCE3's output stage."""
        self.net.eval()
        cv = self._control_vec(c)
        g = self._signal_gain(cv)
        xb = (np.ascontiguousarray(x, np.float32).reshape(-1) * g).astype(np.float32)
        with torch.no_grad():
            o = self.net(torch.from_numpy(xb).to(self.device).view(1, -1)).view(-1).cpu().numpy()
        if self.dcblock_fc > 0:
            from scipy.signal import lfilter

            b, a = self._dc_ba()
            o = lfilter(b, a, o)
        a = float(self.net.out_bound)
        return np.clip(o, -a, a).astype(np.float32)


def held_esr(model: CIRCE3, test: Dataset) -> float:
    s, e = _segments(test.controls)[0]
    x = np.ascontiguousarray(test.x[s:e], np.float32)
    y = np.ascontiguousarray(test.y[s:e], np.float32)
    g = float(test.controls[s, 0])
    pred = model.process(x, np.array([g], np.float32))
    lat = model.latency_samples
    n = min(len(y), len(pred))
    yc, pc = y[:n], pred[:n]
    if lat > 0:
        pc = pc[lat:]
        yc = yc[: len(pc)]
    return float(M.esr(yc[2048:], pc[2048:]))


def _fit_eval(model: CIRCE3, tr: Dataset, ts: Dataset, seed: int) -> float:
    model.fit(tr, ts, TrainConfig(epochs=EPOCHS, lr=3e-3, seq_len=4096,
                                  batch_size=12, warmup=2048, seed=seed))
    return held_esr(model, ts)


def main() -> None:
    log(f"CIRCE-X MoE probe: branches={BRANCHES}, {EPOCHS}ep OS1, vs single-branch mixed TCN")
    results: dict[str, dict] = {}
    for key, sweep_nm, test_nm, kind, nseeds in CIRCUITS:
        tr = Dataset.load(f"data/{sweep_nm}.npz")
        ts = Dataset.load(f"data/{test_nm}.npz")
        row: dict[str, Any] = {"kind": kind}
        # --- baseline: single-branch mixed TCN (the current default), OS1 ---
        bvals = []
        for seed in SEEDS[:nseeds]:
            t = time.time()
            torch.manual_seed(seed)
            base = CIRCE3(n_control=1, signal_idx=(0,), channels=24, n_blocks=2, n_layers=9,
                          oversample=1, device="cuda")
            h = _fit_eval(base, tr, ts, seed)
            bvals.append(h)
            log(f"{key:12s} baseline seed{seed} held {h:.4f} params={base.num_params()} "
                f"({time.time()-t:.0f}s)")
        # --- CIRCE-X: multi-branch MoE, OS1 ---
        xvals = []
        gate_w = None
        xparams = None
        for seed in SEEDS[:nseeds]:
            t = time.time()
            torch.manual_seed(seed)
            mx = CIRCEX(BRANCHES, channels=24, device="cuda")
            h = _fit_eval(mx, tr, ts, seed)
            xvals.append(h)
            if gate_w is None:
                xparams = mx.num_params()
                s, e = _segments(ts.controls)[0]
                gx = float(ts.controls[s, 0])
                xin = torch.from_numpy((ts.x[s:e] * gx).astype(np.float32)).to(mx.device).view(1, -1)
                with torch.no_grad():
                    gate_w = mx.net.gate_weights(xin).cpu().numpy().tolist()
            gstr = [f"{g:.2f}" for g in (gate_w or [])]
            log(f"{key:12s} circex   seed{seed} held {h:.4f} params={mx.num_params()} "
                f"gate={gstr} ({time.time()-t:.0f}s)")
        row.update(baseline=float(np.mean(bvals)), baseline_seeds=bvals,
                   circex=float(np.mean(xvals)), circex_seeds=xvals,
                   gate=gate_w, base_params=base.num_params(), circex_params=xparams)
        results[key] = row
        log(f"  -> {key}: baseline {row['baseline']:.4f}  circex {row['circex']:.4f}  "
            f"gate(long,mixed,shallow)={gate_w}")
    Path("outputs/circex_probe.json").write_text(json.dumps(results, indent=2))
    bmean = float(np.mean([r["baseline"] for r in results.values()]))
    xmean = float(np.mean([r["circex"] for r in results.values()]))
    bworst = max(r["baseline"] for r in results.values())
    xworst = max(r["circex"] for r in results.values())
    log("=== CIRCE-X MoE vs single-branch mixed TCN (held-ESR, OS1) ===")
    for key, _, _, kind, _ in CIRCUITS:
        r = results[key]
        d = 100.0 * (r["circex"] - r["baseline"]) / r["baseline"]
        log(f"  {key:12s} [{kind:6s}] {r['baseline']:.4f} -> {r['circex']:.4f} ({d:+.0f}%)")
    log(f"  MEAN  {bmean:.4f} -> {xmean:.4f}  |  WORST {bworst:.4f} -> {xworst:.4f}")
    log(f"  params: baseline {results['bjt']['base_params']} vs circex {results['bjt']['circex_params']}")
    log("wrote outputs/circex_probe.json")


if __name__ == "__main__":
    main()
