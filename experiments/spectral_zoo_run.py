"""Train the spectral-operator zoo head-to-head + anchor baselines.

Loads every variant Net from experiments/spectral_zoo/, wraps it in CIRCE3's
training machinery (OS1, input-scaling, pre-emph+ESR loss, grad-clip — same harness
as every other probe so the A/B is fair), trains on bjt (strong distortion) + jfet
(mild), and reports held-ESR + high-band(>4k) ESR + params + an offline forward
throughput RTF proxy. Anchored by the time-only mixed-TCN and the time+spectral
hybrid baselines.

The question: can a spectral-mixing + time-nonlinearity operator MATCH/BEAT the
time-domain TCN (bjt 0.029 / jfet 0.0098) more CHEAPLY?

Run: uv run python -m experiments.spectral_zoo_run
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.common import load_pair, make_log
from experiments.spectral_probe import SpectralHybrid, _eval
from vguitar.config import TrainConfig
from vguitar.models.circe3 import CIRCE3

EPOCHS = 150
CIRCUITS = [("bjt", "bjt_bench_sweep", "bjt_bench_fp_test", "strong"),
            ("jfet", "jfet_bench_sweep", "jfet_bench_fp_test", "mild")]
ZOO_DIR = Path("experiments/spectral_zoo")
log = make_log("spectral_zoo_run")


class ZooModel(CIRCE3):
    """Wrap an arbitrary spectral-zoo Net in CIRCE3's harness (OS1 offline eval)."""

    name = "zoo"

    def __init__(self, net: torch.nn.Module, device: str = "cpu") -> None:
        super().__init__(n_control=1, signal_idx=(0,), channels=24,
                         n_blocks=1, n_layers=1, oversample=1, device=device)
        self.net = net.to(self.device)

    process = SpectralHybrid.process  # offline torch forward + 1 Hz DC-block


def _load_net_cls(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"spectral_zoo.{name}", ZOO_DIR / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Net


def _rtf_proxy(model: CIRCE3, sr: int = 44100, dur: float = 2.0) -> float:
    """Offline forward throughput / audio seconds (overestimates true streaming RTF,
    but a fair RELATIVE cost across variants)."""
    x = np.random.default_rng(0).standard_normal(int(sr * dur)).astype(np.float32) * 0.3
    c = np.array([1.0], np.float32)
    model.process(x[:2048], c)  # warm-up
    t0 = time.time()
    model.process(x, c)
    return dur / max(time.time() - t0, 1e-6)


def _build(tag: str, device: str) -> CIRCE3:
    """Build a model for a leaderboard row: baselines or a zoo variant."""
    if tag == "time_only":
        return SpectralHybrid(channels=24, n_layers=9, use_spectral=False, device=device)
    if tag == "hybrid":
        return SpectralHybrid(channels=24, n_layers=9, use_spectral=True, spec_hidden=16,
                              device=device)
    net = _load_net_cls(tag)(channels=24)
    return ZooModel(net, device=device)


def main() -> None:
    zoo = sorted(p.stem for p in ZOO_DIR.glob("*.py") if p.stem != "__init__")
    rows = ["time_only", "hybrid", *zoo]
    log(f"spectral-zoo head-to-head: {len(rows)} models x {len(CIRCUITS)} circuits, {EPOCHS}ep OS1")
    log(f"  models: {rows}")
    results: dict[str, dict] = {}
    rtf: dict[str, float] = {}
    params: dict[str, int] = {}
    for tag in rows:
        results[tag] = {}
        for key, sweep_nm, test_nm, _kind in CIRCUITS:
            tr, ts = load_pair(sweep_nm, test_nm)
            t = time.time()
            torch.manual_seed(0)
            try:
                m = _build(tag, "cuda")
                m.fit(tr, ts, TrainConfig(epochs=EPOCHS, lr=3e-3, seq_len=4096,
                                          batch_size=12, warmup=2048, seed=0))
                esr, band = _eval(m, ts)
                results[tag][key] = {"esr": esr, "band": band}
                params[tag] = m.num_params()
                if tag not in rtf:
                    rtf[tag] = _rtf_proxy(m)
                log(f"{tag:24s} {key:5s} ESR {esr:.4f} band>4k {band:.4f} "
                    f"params={m.num_params()} ({time.time()-t:.0f}s)")
            except Exception as exc:  # one variant must not abort the sweep
                results[tag][key] = {"esr": float("nan"), "band": float("nan"), "error": str(exc)}
                log(f"{tag:24s} {key:5s} FAILED: {exc}")
    out = {"results": results, "rtf_proxy": rtf, "params": params, "epochs": EPOCHS}
    Path("outputs/spectral_zoo.json").write_text(json.dumps(out, indent=2))
    log("=== SPECTRAL-OPERATOR ZOO LEADERBOARD (held-ESR; bjt | jfet) ===")
    ranked = sorted(rows, key=lambda t: np.nanmean(
        [results[t].get(c[0], {}).get("esr", np.nan) for c in CIRCUITS]))
    for tag in ranked:
        b = results[tag].get("bjt", {}).get("esr", float("nan"))
        j = results[tag].get("jfet", {}).get("esr", float("nan"))
        bb = results[tag].get("bjt", {}).get("band", float("nan"))
        log(f"  {tag:24s} bjt {b:.4f} | jfet {j:.4f}  (bjt band>4k {bb:.4f})  "
            f"params={params.get(tag, 0):>7}  rtf~{rtf.get(tag, float('nan')):.1f}x")
    log("wrote outputs/spectral_zoo.json")


if __name__ == "__main__":
    main()
