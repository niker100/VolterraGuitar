"""Structured closed-form probe: a Hammerstein/Volterra delay-embedding + ridge-lstsq
readout. NO training (the "always better than GD" closed-form ideal).

y[t] = sum_{p,tau} a_{p,tau} phi_p(x[t-tau]) is LINEAR in the coefficients a, so it is
solved exactly by lstsq — P static-nonlinear branches (phi_p), each followed by its own
FIR over a set of delays, jointly. This matches the circuits' static-NL + linear-filter
structure far more directly than the random conv features of the ELM (which plateaued
~0.04-0.2). Tests whether a circuit-structured basis closes the gap to the frontier in
closed form. Memory-safe via the elm_probe chunked solve. Short FIR memory (max delay 512
= ~12 ms), so it targets the SHORT-memory (smooth/dead-zone) circuits, not hysteretic.

Run (background): uv run python -m experiments.sota.hammerstein_probe
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.common import make_log
from experiments.sota.elm_probe import REF, _segs, fit_readout, predict
from experiments.sota.harness import CIRCUITS as HC
from vguitar import metrics as M
from vguitar.data import Dataset
from vguitar.models.archive.circe3 import _segments

DEVICE = "cuda"
CIRCUITS = ["bjt", "jfet", "tube_screamer", "crossover", "hard_clipper"]
DELAYS = [*range(0, 33), 40, 48, 64, 96, 128, 192, 256, 384, 512]  # dense short + dilated long


class Hammerstein:
    """P static-nonlinear branches phi_p, each with an FIR over DELAYS (solved by lstsq)."""

    def __init__(self, delays: list[int], device: str = DEVICE) -> None:
        self.delays = delays
        self.device = device
        self.rf = max(delays) + 1
        self.n_feat = 8 * len(delays)

    def _basis(self, x: torch.Tensor) -> torch.Tensor:  # (T,) -> (8, T)
        return torch.stack([
            x, x.abs(), x * x.abs(), torch.tanh(2.0 * x), torch.tanh(6.0 * x),
            F.relu(x - 0.2), F.relu(-x - 0.2), torch.sin(4.0 * x),
        ])

    def features(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.reshape(-1)
        t = xf.shape[0]
        out = [self._basis(F.pad(xf, (d, 0))[:t] if d else xf) for d in self.delays]
        return torch.cat(out, 0)  # (8*len(delays), T)

    def chunk_feats(self, x: torch.Tensor, start: int, n: int) -> torch.Tensor:
        lo = max(0, start - (self.rf - 1))
        return self.features(x[lo:start + n])[:, start - lo:]

    def free(self) -> None:
        pass


def main() -> None:
    log = make_log("sota_hammerstein_probe")
    out = Path("outputs/sota/hammerstein_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    log(f"Hammerstein closed-form: {len(DELAYS)} delays x 8 basis = {8*len(DELAYS)} feats, ridge-lstsq")
    results: dict = {}
    for key in CIRCUITS:
        sweep, test, kind = HC[key]
        tr = Dataset.load(f"data/{sweep}.npz")
        ts = Dataset.load(f"data/{test}.npz")
        segs = _segs(tr)
        s, e = _segments(ts.controls)[0]
        gt = float(ts.controls[s, 0])
        xt = (ts.x[s:e] * gt).astype(np.float32)
        yt = ts.y[s:e].astype(np.float32)
        t = time.time()
        net = Hammerstein(DELAYS)
        w = fit_readout(net, segs, ridge=1e-2)
        pred = predict(net, w, xt)
        n = min(len(yt), len(pred))
        esr = float(M.esr(yt[:n][2048:], pred[:n][2048:]))
        results[key] = {"kind": kind, "held": esr, "n_feat": net.n_feat,
                        "secs": time.time() - t, "ref_trained": REF.get(key)}
        log(f"{key:14s} held={esr:.4f} n_feat={net.n_feat} ({time.time()-t:.1f}s)  "
            f"[trained ref {REF.get(key)}]")
        out.write_text(json.dumps(results, indent=2))
    log("done — wrote outputs/sota/hammerstein_probe.json")


if __name__ == "__main__":
    main()
