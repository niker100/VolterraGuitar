"""ELM / variable-projection probe (user idea): random FIXED dilated-conv features +
a closed-form least-squares readout — no hidden-layer training.

A random dilated causal-conv stack is a random multi-timescale nonlinear temporal
feature basis (reservoir-style); if it is broad enough, the target y(x) lies (near) its
span and the linear readout w is solved optimally by ridge lstsq in ONE shot — training
collapses from minutes to seconds, and it stays streaming-exact + real-time. Signal
control folds into the input (g.x) exactly as in CIRCE3.

Memory-safe: HtH / Hty are accumulated over TIME-CHUNKS (peak memory = one chunk, not the
full width x T matrix), with explicit frees — so the basis width can be pushed broad
("broad enough", per the idea) without OOM. Readout is plain ridge L2 = the exact ESR
objective (we report plain held-ESR). Causal features per chunk use rf-1 left context.

OFFLINE accuracy gauge (base rate). Compares to trained-CIRCE3 refs. Knobs: width, depth,
init scale, ridge. Run (background): uv run python -m experiments.sota.elm_probe
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.common import make_log
from experiments.sota.harness import CIRCUITS as HC
from vguitar import metrics as M
from vguitar.data import Dataset
from vguitar.models.circe3 import _segments

DEVICE = "cuda"
CHUNK = 32768
CIRCUITS = ["bjt", "jfet", "tube_screamer", "hard_clipper"]
REF = {"bjt": 0.0046, "jfet": 0.0020, "tube_screamer": 0.0145, "crossover": 0.0225,
       "hard_clipper": 0.0558}


class RandomFeatures:
    """Fixed random dilated causal-conv stack -> concatenated per-layer activations
    (tanh / relu / sin groups). Weights drawn once (seeded), never trained;
    ``init_scale`` is the "broad enough" knob."""

    def __init__(self, width: int = 128, n_layers: int = 8, init_scale: float = 1.0,
                 kernel: int = 3, seed: int = 0, device: str = DEVICE) -> None:
        g = torch.Generator().manual_seed(seed)
        self.width, self.n_layers, self.kernel = width, n_layers, kernel
        self.device = device
        self.dils = [2**i for i in range(n_layers)]
        self.rf = 1 + (kernel - 1) * sum(self.dils)  # causal receptive field (samples)
        sc = init_scale
        self.win = (torch.randn(width, 1, 1, generator=g) * sc).to(device)
        self.bin = (torch.randn(width, generator=g) * 0.1).to(device)
        self.w = [(torch.randn(width, width, kernel, generator=g) * sc / np.sqrt(width)).to(device)
                  for _ in range(n_layers)]
        self.b = [(torch.randn(width, generator=g) * 0.1).to(device) for _ in range(n_layers)]
        self.n_feat = width * (n_layers + 1)

    def _act(self, h: torch.Tensor) -> torch.Tensor:
        n = h.shape[0] // 3
        return torch.cat([torch.tanh(h[:n]), F.relu(h[n:2 * n]), torch.sin(h[2 * n:])], 0)

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` (T,) -> features (n_feat, T), causal (left zero-pad)."""
        h = self._act(F.conv1d(x.view(1, 1, -1), self.win, self.bin).squeeze(0))
        feats = [h]
        for w, b, d in zip(self.w, self.b, self.dils, strict=True):
            hp = F.pad(h.unsqueeze(0), ((self.kernel - 1) * d, 0))
            h = self._act(F.conv1d(hp, w, b, dilation=d).squeeze(0))
            feats.append(h)
        return torch.cat(feats, 0)

    @torch.no_grad()
    def chunk_feats(self, x: torch.Tensor, start: int, n: int) -> torch.Tensor:
        """Causal features for x[start:start+n], using rf-1 real left context (so the
        result is identical to full-signal features but chunk-bounded in memory)."""
        lo = max(0, start - (self.rf - 1))
        h = self.features(x[lo:start + n])
        return h[:, start - lo:]

    def free(self) -> None:
        del self.win, self.bin, self.w, self.b
        torch.cuda.empty_cache()


def _segs(ds: Dataset) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for s, e in _segments(ds.controls):
        g = float(ds.controls[s, 0])
        out.append(((ds.x[s:e] * g).astype(np.float32), ds.y[s:e].astype(np.float32)))
    return out


@torch.no_grad()
def fit_readout(net: RandomFeatures, segs: list[tuple[np.ndarray, np.ndarray]],
                ridge: float, chunk: int = CHUNK) -> torch.Tensor:
    """Ridge lstsq readout w minimizing ||y - w.[H;1]||^2 (plain L2 = the ESR objective),
    accumulated chunk-by-chunk in float64 — peak memory is one chunk of features."""
    nf = net.n_feat + 1
    a = torch.zeros(nf, nf, device=net.device, dtype=torch.float64)
    bv = torch.zeros(nf, device=net.device, dtype=torch.float64)
    for xb, yb in segs:
        x = torch.from_numpy(xb).to(net.device)
        y = torch.from_numpy(yb).to(net.device)
        for start in range(0, x.shape[0], chunk):
            n = min(chunk, x.shape[0] - start)
            h = net.chunk_feats(x, start, n)
            hb = torch.cat([h, torch.ones(1, n, device=net.device)], 0).double()
            a += hb @ hb.T
            bv += hb @ y[start:start + n].double()
            del h, hb
        del x, y
        torch.cuda.empty_cache()
    a += ridge * torch.eye(nf, device=net.device, dtype=torch.float64)
    return torch.linalg.solve(a, bv).float()


@torch.no_grad()
def predict(net: RandomFeatures, w: torch.Tensor, x: np.ndarray, chunk: int = CHUNK) -> np.ndarray:
    xt = torch.from_numpy(x.astype(np.float32)).to(net.device)
    out = torch.empty(xt.shape[0], device=net.device)
    for start in range(0, xt.shape[0], chunk):
        n = min(chunk, xt.shape[0] - start)
        h = net.chunk_feats(xt, start, n)
        hb = torch.cat([h, torch.ones(1, n, device=net.device)], 0)
        out[start:start + n] = w @ hb
        del h, hb
    return out.cpu().numpy()


def main() -> None:
    log = make_log("sota_elm_probe")
    out = Path("outputs/sota/elm_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    # clean width ladder + init scale (now memory-safe -> push the basis broad)
    configs = [
        {"label": "w256_s0.5", "width": 256, "init_scale": 0.5},
        {"label": "w256_s1.0", "width": 256, "init_scale": 1.0},
        {"label": "w512_s0.5", "width": 512, "init_scale": 0.5},
        {"label": "w512_s1.0", "width": 512, "init_scale": 1.0},
        {"label": "w1024_s0.5", "width": 1024, "init_scale": 0.5},
    ]
    log(f"ELM probe (memory-safe, chunk={CHUNK}, plain-L2): {[c['label'] for c in configs]}")
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
        row: dict = {"kind": kind, "ref_trained": REF.get(key)}
        for cfg in configs:
            t = time.time()
            net = RandomFeatures(cfg["width"], 8, cfg["init_scale"], seed=0)
            w = fit_readout(net, segs, ridge=1e-2)
            pred = predict(net, w, xt)
            n = min(len(yt), len(pred))
            esr = float(M.esr(yt[:n][2048:], pred[:n][2048:]))
            row[cfg["label"]] = {"held": esr, "n_feat": net.n_feat, "secs": time.time() - t}
            log(f"{key:14s} {cfg['label']:11s} held={esr:.4f} n_feat={net.n_feat} "
                f"({time.time()-t:.1f}s)  [trained ref {REF.get(key)}]")
            net.free()
            del net, w
            torch.cuda.empty_cache()
        results[key] = row
        out.write_text(json.dumps(results, indent=2))
    log("done — wrote outputs/sota/elm_probe.json")


if __name__ == "__main__":
    main()
