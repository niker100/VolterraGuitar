"""Step 3 escalation â€” does the spectral-branch formant win survive a SLIM net?

The prototype's spectral MLP was 422k params (513->256->1026); the win must survive
a much smaller net to be real-time-affordable. Sweep the spectral hidden width on
the two clearest winners (bjt, jfet) and read whether band>4k stays low as params
fall. If it holds at small hidden, the causal overlap-add streaming build is worth
it (RTF will be fine); if it needs the big MLP, real-time is doubtful.

Run: uv run python -m experiments.spectral_slim
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from experiments.archive.spectral_probe import SpectralHybrid, _eval
from experiments.common import load_pair, make_log
from vguitar.config import TrainConfig

EPOCHS = 150
CIRCUITS = [("bjt", "bjt_bench_sweep", "bjt_bench_fp_test"),
            ("jfet", "jfet_bench_sweep", "jfet_bench_fp_test")]
HIDDENS = [0, 16, 32, 64, 256]  # 0 = time-only baseline (no spectral branch)
log = make_log("spectral_slim")


def main() -> None:
    log(f"spectral slim sweep: hidden in {HIDDENS} (0=time-only), {EPOCHS}ep OS1")
    results: dict[str, dict] = {}
    for key, sweep_nm, test_nm in CIRCUITS:
        tr, ts = load_pair(sweep_nm, test_nm)
        results[key] = {}
        for hid in HIDDENS:
            t = time.time()
            torch.manual_seed(0)
            use = hid > 0
            m = SpectralHybrid(channels=24, n_layers=9, use_spectral=use,
                               spec_hidden=max(hid, 1), device="cuda")
            m.fit(tr, ts, TrainConfig(epochs=EPOCHS, lr=3e-3, seq_len=4096,
                                      batch_size=12, warmup=2048, seed=0))
            esr, band = _eval(m, ts)
            p = m.num_params()
            results[key][f"h{hid}"] = {"hidden": hid, "esr": esr, "band": band, "params": p}
            log(f"{key:6s} hidden={hid:4d} ESR {esr:.4f} band>4k {band:.4f} params={p} "
                f"({time.time()-t:.0f}s)")
    Path("outputs/spectral_slim.json").write_text(json.dumps(results, indent=2))
    log("=== SPECTRAL SLIM SWEEP (does the band>4k win survive small hidden?) ===")
    for key, _, _ in CIRCUITS:
        base = results[key]["h0"]["band"]
        for hid in HIDDENS:
            r = results[key][f"h{hid}"]
            d = 100.0 * (r["band"] - base) / base if base else 0.0
            tag = "(time-only)" if hid == 0 else f"({d:+.0f}% band vs time-only)"
            log(f"  {key:6s} h{hid:<4d} ESR {r['esr']:.4f} band {r['band']:.4f} {tag} "
                f"params={r['params']}")
    log("wrote outputs/spectral_slim.json")


if __name__ == "__main__":
    main()
