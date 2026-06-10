"""Step 3 — spectral-domain (STFT) hybrid prototype: does a complex spectral branch
improve formant/high-band fidelity over a pure time-domain TCN?

Motivation (frontier-plan Step 3): these circuits are linear-filter ∘ instantaneous
nonlinearity ∘ linear-filter. The LINEAR parts are long convolutions = a complex
multiply per STFT bin (the formant/transfer structure a time-domain TCN smooths);
the NONLINEAR part is time-local waveshaping (a small time head). So:

    y = time_head(x)  +  ISTFT( G(|STFT(x)|) ⊙ STFT(x) )

The spectral gain ``G`` is predicted per-frame from the frame MAGNITUDE (an MLP),
so the spectral path is **input-dependent** — a learned time-varying filter, not a
static LTI mask (a static mask cannot create harmonics; the harmonics still come
from the time head, the spectral path carries the linear formant/memory structure).

This PROTOTYPE is offline torch (``center=True`` STFT, non-causal — we are testing
whether the REPRESENTATION helps before paying for causal overlap-add streaming).
A/B: spectral branch ON vs OFF (pure time head), same harness (reuses CIRCE3.fit),
OS1, on the formant-rich smooth circuits (bjt/jfet/TS) + a hard one (crossover).
Reports overall held-ESR AND high-band (>4 kHz) ESR — the spectral-fidelity metric
this branch exists to move. If it improves the high band without regressing overall
or RTF, escalate to multi-length STFT + a causal streaming version; else null.

Run: uv run python -m experiments.spectral_probe
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from experiments.common import load_pair, make_log
from vguitar.config import TrainConfig
from vguitar.models.archive.circe3 import CIRCE3, _segments
from vguitar.models.archive.tcn import _MixedLayer

EPOCHS = 150
N_FFT = 1024
HOP = 256
CIRCUITS = [
    ("bjt", "bjt_bench_sweep", "bjt_bench_fp_test", "smooth", 1),
    ("jfet", "jfet_bench_sweep", "jfet_bench_fp_test", "smooth", 1),
    ("tube_screamer", "tube_screamer_bench_sweep", "tube_screamer_bench_fp_test", "smooth", 1),
    ("crossover", "crossover_classb_edge_sweep", "crossover_classb_edge_test", "hard", 2),
]
SEEDS = (0, 7)
log = make_log("spectral_probe")


class _SpectralHybridNet(nn.Module):
    """Time-domain TCN waveshaper + an optional input-dependent complex spectral branch."""

    out_bound: torch.Tensor
    window: torch.Tensor

    def __init__(self, channels: int, n_layers: int, kernel: int = 3,
                 n_fft: int = N_FFT, hop: int = HOP, use_spectral: bool = True,
                 spec_hidden: int = 256, use_time: bool = True) -> None:
        super().__init__()
        self.use_spectral = use_spectral
        self.use_time = use_time
        self.n_fft, self.hop = n_fft, hop
        self.spec_hidden = spec_hidden
        # --- time head: a mixed-activation TCN (the waveshaper) ---
        if use_time:
            self.input = nn.Conv1d(1, channels, 1)
            self.layers = nn.ModuleList(
                _MixedLayer(channels, kernel, 2**i) for i in range(n_layers)
            )
            self.head = nn.Sequential(
                nn.ReLU(), nn.Conv1d(channels, channels, 1), nn.ReLU(), nn.Conv1d(channels, 1, 1)
            )
        # --- spectral branch: per-frame complex gain predicted from the magnitude ---
        if use_spectral:
            nbins = n_fft // 2 + 1
            hidden = spec_hidden
            self.spec = nn.Sequential(
                nn.Linear(nbins, hidden), nn.ReLU(), nn.Linear(hidden, 2 * nbins)
            )
            # zero-init the readout so the spectral branch starts as a near-pass-through
            # gain of (1+0j): G = 1 + (gr + i*gi), gr=gi=0 at init -> identity, gradient flows.
            nn.init.zeros_(self.spec[-1].weight)
            nn.init.zeros_(self.spec[-1].bias)
            self.register_buffer("window", torch.hann_window(n_fft))
        self.register_buffer("out_bound", torch.tensor(1.0))

    def _time(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x.unsqueeze(1))
        skip = torch.zeros_like(h)
        for layer in self.layers:
            h, sk = layer(h)
            skip = skip + sk
        return self.head(skip).squeeze(1)

    def _spectral(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        X = torch.stft(x, self.n_fft, self.hop, window=self.window, center=True,
                       return_complex=True)  # (B, F, frames)
        mag = X.abs().transpose(1, 2)  # (B, frames, F)
        g = self.spec(mag)  # (B, frames, 2F)
        nb = X.shape[1]
        gr, gi = g[..., :nb], g[..., nb:]
        gain = torch.complex(1.0 + gr, gi).transpose(1, 2)  # (B, F, frames), ~1 at init
        y = torch.istft(X * gain, self.n_fft, self.hop, window=self.window, center=True, length=n)
        return y

    def raw(self, x: torch.Tensor, c_sys: torch.Tensor | None = None) -> torch.Tensor:
        y = self._time(x) if self.use_time else None
        if self.use_spectral:
            s = self._spectral(x)
            y = s if y is None else y + s
        assert y is not None, "need at least one of use_time / use_spectral"
        return y

    def forward(self, x: torch.Tensor, c_sys: torch.Tensor | None = None) -> torch.Tensor:
        a = float(self.out_bound)
        return torch.clamp(self.raw(x, c_sys), -a, a)


class SpectralHybrid(CIRCE3):
    """Spectral+time hybrid over CIRCE3's machinery; OS1 offline eval (prototype)."""

    name = "spectral_hybrid"

    def __init__(self, channels: int = 24, n_layers: int = 9, use_spectral: bool = True,
                 spec_hidden: int = 256, use_time: bool = True, device: str = "cpu",
                 **kw: Any) -> None:
        super().__init__(n_control=1, signal_idx=(0,), channels=channels,
                         n_blocks=1, n_layers=1, oversample=1, device=device, **kw)
        self.use_spectral = use_spectral
        self._nl = n_layers
        self.net = _SpectralHybridNet(channels, n_layers, use_spectral=use_spectral,
                                      spec_hidden=spec_hidden, use_time=use_time).to(self.device)

    def process(self, x: np.ndarray, c: np.ndarray | None = None) -> np.ndarray:
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


def _band_esr(y: np.ndarray, yhat: np.ndarray, sr: int = 44100, lo: float = 4000.0) -> float:
    """ESR restricted to the >lo Hz band (the spectral/formant-fidelity proxy)."""
    n = min(len(y), len(yhat))
    Y = np.fft.rfft(y[:n])
    P = np.fft.rfft(yhat[:n])
    f = np.fft.rfftfreq(n, 1.0 / sr)
    m = f >= lo
    num = float(np.sum(np.abs((Y - P)[m]) ** 2))
    den = float(np.sum(np.abs(Y[m]) ** 2)) + 1e-12
    return num / den


def _eval(model: CIRCE3, test: Any) -> tuple[float, float]:
    from vguitar import metrics as M

    s, e = _segments(test.controls)[0]
    x = np.ascontiguousarray(test.x[s:e], np.float32)
    y = np.ascontiguousarray(test.y[s:e], np.float32)
    g = float(test.controls[s, 0])
    pred = np.asarray(model.process(x, np.array([g], np.float32)), np.float32)
    n = min(len(y), len(pred))
    yc, pc = y[:n][2048:], pred[:n][2048:]
    return float(M.esr(yc, pc)), _band_esr(yc, pc, test.sr)


def main() -> None:
    log(f"spectral-hybrid A/B: n_fft={N_FFT} hop={HOP}, {EPOCHS}ep OS1, spectral ON vs OFF")
    results: dict[str, dict] = {}
    for key, sweep_nm, test_nm, kind, nseeds in CIRCUITS:
        tr, ts = load_pair(sweep_nm, test_nm)
        row: dict[str, Any] = {"kind": kind}
        for tag, use_spec in [("time_only", False), ("hybrid", True)]:
            esrs, bands = [], []
            params = None
            for seed in SEEDS[:nseeds]:
                t = time.time()
                torch.manual_seed(seed)
                m = SpectralHybrid(channels=24, n_layers=9, use_spectral=use_spec, device="cuda")
                m.fit(tr, ts, TrainConfig(epochs=EPOCHS, lr=3e-3, seq_len=4096,
                                          batch_size=12, warmup=2048, seed=seed))
                esr, band = _eval(m, ts)
                esrs.append(esr)
                bands.append(band)
                params = m.num_params()
                log(f"{key:12s} {tag:9s} seed{seed} ESR {esr:.4f} band>4k {band:.4f} "
                    f"params={params} ({time.time()-t:.0f}s)")
            row[tag] = {"esr": float(np.mean(esrs)), "band": float(np.mean(bands)),
                        "esr_seeds": esrs, "params": params}
        results[key] = row
        de = 100.0 * (row["hybrid"]["esr"] - row["time_only"]["esr"]) / row["time_only"]["esr"]
        db = 100.0 * (row["hybrid"]["band"] - row["time_only"]["band"]) / (row["time_only"]["band"] + 1e-9)
        log(f"  -> {key}: ESR {row['time_only']['esr']:.4f}->{row['hybrid']['esr']:.4f} ({de:+.0f}%)  "
            f"band>4k {row['time_only']['band']:.4f}->{row['hybrid']['band']:.4f} ({db:+.0f}%)")
    Path("outputs/spectral_probe.json").write_text(json.dumps(results, indent=2))
    log("=== SPECTRAL-HYBRID A/B (time-only -> +spectral; ESR / band>4k) ===")
    for key, _, _, kind, _ in CIRCUITS:
        r = results[key]
        log(f"  {key:12s} [{kind:6s}] ESR {r['time_only']['esr']:.4f}->{r['hybrid']['esr']:.4f}  "
            f"band>4k {r['time_only']['band']:.4f}->{r['hybrid']['band']:.4f}")
    log("wrote outputs/spectral_probe.json")


if __name__ == "__main__":
    main()
