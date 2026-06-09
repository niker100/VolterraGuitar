"""CIRCE3 — the SOTA real-time emulator of nonlinear systems with exogenous controls.

This is the clean, final architecture, derived from approximation + information
theory and confirmed by experiment (see ``docs/optimal-architecture.md`` for the
derivation and the decisive A/B). It is one idea, applied honestly:

**A well-trained dilated causal TCN** — the Boyd-Chua canonical realizer of a
causal, time-invariant, fading-memory operator, near-optimal in the
minimal-description-length sense and unbeaten by any alternative backbone — with
exogenous controls handled by their **physical kind**:

* **Signal-acting** controls (drive, gain, sustain, level, a forcing amplitude)
  are folded **directly into the input** as a gain, ``x <- g.x``. The model is
  *unconditioned* on them, so it reproduces any setting *exactly by construction*
  (interpolation and extrapolation) with **zero** conditioning parameters and no
  interpolation error — a 0-bit equivariance a learned conditioner can only spoil.
* **System-acting** controls (a tone-stack component, a bias, a nonlinearity
  coefficient) drive a **minimal per-block FiLM** ``h <- gamma(c).h + beta(c)``
  from a tiny zero-init conditioner. Per-layer FiLM across a deep TCN reaches both
  static-map and dynamics-changing controls; in the A/B it matched or beat every
  heavier conditioner (concat, rational, Chebyshev, Fourier, hypernetwork), which
  were all retired.

Each TCN layer's nonlinearity is a **heterogeneous mixed activation** by default
(``block_act='mixed'``): the channels are split into five groups carrying tanh,
gelu, relu, abs and Snake units, so every layer has corner-capable
(abs/relu/Snake) units to synthesize the slope discontinuities of sharp circuits
*and* smooth units the layer's ``1x1`` recombine can favour where corners are not
needed. Under a fixed gradient-norm clip (which removes its earlier training
instability) the mixed block is uniformly at-least-as-good as the classic WaveNet
``tanh*sigmoid`` gate across the circuit battery — it wins the hardest circuits
(class-B crossover -13%, wavefolder -8%, fullwave -7%) at a mild, accepted cost on
the smoothest (JFET +14%, tube-screamer +3%) — while being **smaller and faster**
(54k params / 2.45x RTF vs the gate's 85k / 2.23x, the ``ch->ch`` conv replacing
the gate's ``ch->2ch``). The classic gate is kept as ``block_act='gated'``.

Output stage: a clamp saturator of last resort + a fixed 1-pole 1 Hz DC-blocker
(low corner: removes only true DC — the asymmetric stages' ``f(0) != 0`` silence
offset — without the audible-band phase lag a higher corner would add, which
otherwise shifts the static transfer curve and widens its hysteresis loop).
Training: ESR + a **phase-aware pre-emphasis-ESR** term (a first-order high-pass
on the error, Wright & Valimaki ICASSP-20), optimised with Adam under a fixed
**global gradient-norm clip** (``grad_clip=1.0`` by default). The clip is decisive
on sharp-discontinuity circuits: the dead-zone/fold corner produces large,
spiky gradients that, unclipped, knock the optimiser into a far worse minimum and
make the result swing wildly with the seed (class-B crossover held-ESR 0.16-0.39
across seeds, mean 0.27); clipping collapses both the mean and the variance
(0.031-0.035 across seeds, mean 0.033 -- an 8x reduction) and *also* helps the
smooth circuits (BJT -25%, JFET -10%), with zero inference cost. Plain ESR is
energy-dominated and
barely sees the low-energy upper harmonics, so it smooths the circuit's resonant
**harmonic formants** (peaks/notches) and rounds the static transfer curve; the
pre-emphasis term lifts that band into the gradient *without discarding phase*
(unlike a magnitude-STFT term, which trades the transfer curve away), sharpening
the formants AND the transfer curve at once. Cosine LR annealing, GPU.
Streaming is the Fast-WaveNet cached incremental convolution + per-block FiLM, all
numpy: ``process`` (offline) == ``process_block`` (streamed), bit-exact and
block-size invariant even under a moving knob; ``latency_samples == 0``.

On realistic (band-limited) signals this reaches **ESR < 0.005 real-time on every
tested control kind** — signal (BJT drive), static-map (Duffing beta), dynamics
(JFET tone) — the headline result in ``docs/CIRCE3-modelcard.md``. The dominant
lever for held-out accuracy is **control-grid density** (``ESR ~ spacing^k``), not
the conditioner.

With no system controls (``n_system == 0``, e.g. a single drive knob) CIRCE3 is the
unconditioned input-scaling TCN — the minimal form.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from torch import nn

from vguitar.config import TrainConfig
from vguitar.losses import esr_loss, multi_stft_loss, preemph_esr_loss
from vguitar.models.base import (
    FitReport,
    Model,
    pick_device,
    register_model,
    to_inference_cpu,
)
from vguitar.models.tcn import _GatedLayer, _mixed_sizes, _MixedLayer

if TYPE_CHECKING:
    from vguitar.data import Dataset


class _OverSampler:
    """Streaming-exact integer-factor polyphase resampler (numpy).

    A pair of linear-phase windowed-sinc FIR low-passes (cutoff = Nyquist/factor)
    with carried filter state, so block-by-block ``up``/``down`` is **bit-exact**
    with processing the whole signal at once (verified) and block-size invariant.
    Used to run the network at an internal oversampled rate, which removes the
    in-band aliasing the tanh/sigmoid gates would otherwise fold back from above
    Nyquist — the dominant spectral error on band-limited targets. Net group delay
    is ``(taps - 1) // factor`` base-rate samples (the model's ``latency_samples``).
    """

    def __init__(self, factor: int, taps: int = 127) -> None:
        from scipy.signal import firwin

        self.factor = int(factor)
        self.taps = int(taps)
        h = firwin(self.taps, 1.0 / self.factor).astype(np.float32)
        self.up_h = (h * self.factor).astype(np.float32)
        self.down_h = h
        self.reset()

    def reset(self) -> None:
        self._us = np.zeros(self.taps - 1, dtype=np.float32)
        self._ds = np.zeros(self.taps - 1, dtype=np.float32)

    def up(self, x: np.ndarray) -> np.ndarray:
        """Base-rate ``(N,)`` -> oversampled ``(factor*N,)`` (zero-stuff + FIR)."""
        z = np.zeros(x.shape[0] * self.factor, dtype=np.float32)
        z[:: self.factor] = x
        buf = np.concatenate([self._us, z])
        y = np.convolve(buf, self.up_h, mode="valid")
        self._us = buf[-(self.taps - 1) :].copy()
        return y.astype(np.float32)

    def down(self, u: np.ndarray) -> np.ndarray:
        """Oversampled ``(factor*N,)`` -> base-rate ``(N,)`` (FIR + decimate)."""
        buf = np.concatenate([self._ds, u])
        y = np.convolve(buf, self.down_h, mode="valid")
        self._ds = buf[-(self.taps - 1) :].copy()
        return y[:: self.factor].astype(np.float32)


def _os_resample(x: np.ndarray, factor: int, taps: int, *, up: bool) -> np.ndarray:
    """One-shot oversample helper (fresh state) for offline data preparation."""
    os = _OverSampler(factor, taps)
    return os.up(x) if up else os.down(x)


def _n_rect_feats(thr: tuple[float, ...]) -> int:
    """Number of input feature channels for rectified-feature thresholds ``thr``."""
    return 1 + 2 * len(thr) + 1 if thr else 1


def _rect_feats_torch(x: torch.Tensor, thr: tuple[float, ...]) -> torch.Tensor:
    """``(B, T)`` -> ``(B, n_feat, T)`` rectified input-feature stack.

    ``[x] + [relu(x-t), relu(-x-t) for t in thr] + [abs(x)]``. ReLU/abs carry an
    *exact* slope discontinuity a smooth (Lipschitz) net cannot synthesize, so the
    TCN only has to learn smooth dynamics *around* the kink (dead-zone / crossover /
    wavefolder corners). All maps are positive-homogeneous in the already-scaled
    input, so input-scaling equivariance is preserved; the thresholds sit at fixed
    circuit-input volts (where a dead-zone / diode knee lives)."""
    feats = [x]
    for t in thr:
        feats.append(torch.relu(x - t))
        feats.append(torch.relu(-x - t))
    feats.append(torch.abs(x))
    return torch.stack(feats, dim=1)


def _rect_feats_np(x: np.ndarray, thr: tuple[float, ...]) -> np.ndarray:
    """Numpy twin of :func:`_rect_feats_torch`: ``(T,)`` -> ``(n_feat, T)``."""
    feats = [x]
    for t in thr:
        feats.append(np.maximum(x - t, 0.0))
        feats.append(np.maximum(-x - t, 0.0))
    feats.append(np.abs(x))
    return np.stack(feats, axis=0).astype(np.float32)


def _onepole_scan(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Stable log-depth parallel scan of the leaky integrator
    ``s_k[n] = a_k s_k[n-1] + (1 - a_k) x[n]`` (zero initial state).

    ``x`` (B, T), ``a`` (K,) in (0, 1) -> ``s`` (B, K, T). A Hillis-Steele inclusive
    scan: after the step at offset d, ``s[n]`` holds up to 2d terms of the
    exponentially-weighted sum; doubling d reaches the full window in ceil(log2 T)
    steps. Gives the TCN unbounded (O(1)/sample) memory the dilated-FIR receptive
    field lacks; the numpy twin streams the identical recurrence via ``lfilter``
    with carried state, so ``process == process_block`` stays bit-exact."""
    t = x.shape[-1]
    s = (1.0 - a)[None, :, None] * x[:, None, :]  # (B, K, T) input term
    shift, a_pow = 1, a.clone()  # a_pow tracks a^shift
    while shift < t:
        tail = s[..., shift:] + a_pow[None, :, None] * s[..., :-shift]
        s = torch.cat([s[..., :shift], tail], dim=-1)
        shift *= 2
        a_pow = a_pow * a_pow
    return s


def _mixed_act_np(
    conv: np.ndarray, sizes: tuple[int, int, int, int, int], alpha: np.ndarray
) -> np.ndarray:
    """Numpy twin of :class:`vguitar.models.tcn._MixedActivation` on ``(C, T)``.

    Splits the channels into the five activation groups (tanh, gelu, relu, abs,
    snake) and applies each, matching the torch forward bit-for-bit (gelu via the
    exact erf form ``0.5 x (1 + erf(x/sqrt 2))``; snake alpha clamped >= 0.1)."""
    from scipy.special import erf

    i0 = sizes[0]
    i1 = i0 + sizes[1]
    i2 = i1 + sizes[2]
    i3 = i2 + sizes[3]
    x = conv[i0:i1]
    sn = conv[i3:]
    al = np.maximum(alpha, 0.1)[:, None]
    return np.concatenate(
        [
            np.tanh(conv[:i0]),
            0.5 * x * (1.0 + erf(x / np.sqrt(2.0))),
            np.maximum(conv[i1:i2], 0.0),
            np.abs(conv[i2:i3]),
            sn + np.sin(al * sn) ** 2 / al,
        ],
        axis=0,
    ).astype(np.float32)


class _CIRCE3Net(nn.Module):
    """Gated-TCN spine + an optional minimal FiLM on the system controls.

    Signal controls are folded into the input upstream (in :class:`CIRCE3`), so
    this net only sees the system controls. With ``n_system == 0`` there is no
    conditioner and this is a plain TCN.
    """

    c_mean: torch.Tensor
    c_std: torch.Tensor
    out_bound: torch.Tensor

    def __init__(
        self,
        channels: int,
        n_blocks: int,
        n_layers: int,
        kernel: int,
        n_system: int,
        cond_hidden: int,
        rect_thr: tuple[float, ...] = (),
        block_act: str = "mixed",
        out_shaper: str = "none",
        shaper_k: int = 8,
        n_state: int = 0,
        sr_state: int = 44_100,
    ) -> None:
        super().__init__()
        self.n_total = n_blocks * n_layers
        self.channels = channels
        self.n_system = n_system
        self.rect_thr = rect_thr
        self.block_act = block_act
        self.out_shaper = out_shaper
        self.shaper_k = shaper_k
        self.n_state = n_state
        self.input = nn.Conv1d(_n_rect_feats(rect_thr) + n_state, channels, 1)
        # Leaky-integrator state channels: K learnable one-poles fed to the input
        # alongside x, giving unbounded memory the FIR stack lacks. tau init spans
        # ~5..500 ms at the internal (oversampled) rate; a_k = sigmoid(logit) stays in
        # (0,1). Linear in the gain-scaled input -> input-scaling-equivariant.
        if n_state > 0:
            taus = torch.logspace(float(np.log10(5e-3)), float(np.log10(0.5)), n_state)
            a0 = torch.exp(-1.0 / (taus * sr_state)).clamp(1e-4, 1 - 1e-6)
            self.a_logit = nn.Parameter(torch.log(a0 / (1.0 - a0)))
        dilations = [2**i for i in range(n_layers)]
        layer_cls = _MixedLayer if block_act == "mixed" else _GatedLayer
        self.layers = nn.ModuleList(
            layer_cls(channels, kernel, d) for _ in range(n_blocks) for d in dilations
        )
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv1d(channels, 1, 1),
        )
        # Learned periodic (Fourier) waveshaper head: a residual correction
        # y = o + sum_k c_k sin(k.w.o), an explicit multi-fold primitive a smooth
        # TCN cannot synthesize. Zero-init c_k => identity at init (backward-compatible
        # + input-scaling-preserving); pointwise => streaming-exact. w is a learned
        # base angular frequency shared across harmonics.
        if out_shaper == "fourier":
            self.shaper_c = nn.Parameter(torch.zeros(shaper_k))
            self.shaper_w = nn.Parameter(torch.ones(1))
        # Minimal FiLM conditioner on the system controls; zero-init readout so an
        # untrained net is the plain spine (gamma=1, beta=0).
        if n_system > 0:
            cond_out = nn.Linear(cond_hidden, 2 * self.n_total * channels)
            nn.init.zeros_(cond_out.weight)
            nn.init.zeros_(cond_out.bias)
            self.cond = nn.Sequential(nn.Linear(n_system, cond_hidden), nn.Tanh(), cond_out)
            self.register_buffer("c_mean", torch.zeros(n_system))
            self.register_buffer("c_std", torch.ones(n_system))
        self.register_buffer("out_bound", torch.tensor(1.0))

    def film(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """System control ``c`` (B, n_system) -> per-layer ``gamma, beta`` (B, L, C)."""
        cn = (c - self.c_mean) / self.c_std
        gb = self.cond(cn).view(-1, self.n_total, 2, self.channels)
        return 1.0 + torch.tanh(gb[:, :, 0, :]), gb[:, :, 1, :]

    def raw(self, x: torch.Tensor, c_sys: torch.Tensor | None) -> torch.Tensor:
        """Pre-clamp output ``(B, T)``. ``x`` is ALREADY signal-scaled."""
        xin = _rect_feats_torch(x, self.rect_thr) if self.rect_thr else x.unsqueeze(1)
        if self.n_state > 0:
            xin = torch.cat([xin, _onepole_scan(x, torch.sigmoid(self.a_logit))], dim=1)
        h = self.input(xin)  # (B, C, T)
        skip = h.new_zeros(h.shape)
        gamma = beta = None
        if self.n_system > 0:
            assert c_sys is not None
            gamma, beta = self.film(c_sys)
        for i, layer in enumerate(self.layers):
            if gamma is not None and beta is not None:
                h = gamma[:, i, :, None] * h + beta[:, i, :, None]  # FiLM before the layer
            h, sk = layer(h)
            skip = skip + sk
        o = self.out(skip).squeeze(1)
        if self.out_shaper == "fourier":
            k = torch.arange(1, self.shaper_k + 1, device=o.device, dtype=o.dtype)
            o = o + (self.shaper_c * torch.sin(k * self.shaper_w * o.unsqueeze(-1))).sum(-1)
        return o

    def forward(self, x: torch.Tensor, c_sys: torch.Tensor | None) -> torch.Tensor:
        a = float(self.out_bound)
        return torch.clamp(self.raw(x, c_sys), -a, a)


@register_model
class CIRCE3(Model):
    """SOTA conditioned emulator: input-scaling (signal) + minimal FiLM (system)."""

    name: ClassVar[str] = "circe3"
    description: ClassVar[str] = (
        "SOTA real-time emulator: TCN + input-scaling (signal controls) + minimal "
        "FiLM (system controls)."
    )
    latency_samples: int = 0
    conditioned: ClassVar[bool] = True

    def __init__(
        self,
        n_control: int = 1,
        signal_idx: tuple[int, ...] = (0,),
        channels: int = 24,
        n_blocks: int = 2,
        n_layers: int = 8,
        kernel: int = 3,
        cond_hidden: int = 16,
        dcblock_fc: float = 1.0,
        saturator: str = "clamp",
        stft_weight: float = 0.0,
        preemph_weight: float = 1.0,
        preemph_alpha: float = 0.95,
        preemph_order: int = 1,
        oversample: int = 1,
        os_taps: int = 127,
        rect_thr: tuple[float, ...] = (),
        block_act: str = "mixed",
        out_shaper: str = "none",
        shaper_k: int = 8,
        grad_clip: float = 1.0,
        n_state: int = 0,
        device: str = "cpu",
    ) -> None:
        if saturator not in ("clamp", "adaa1", "adaa2"):
            raise ValueError(f"saturator must be 'clamp', 'adaa1' or 'adaa2', got {saturator!r}")
        if block_act not in ("gated", "mixed"):
            raise ValueError(f"block_act must be 'gated' or 'mixed', got {block_act!r}")
        if out_shaper not in ("none", "fourier"):
            raise ValueError(f"out_shaper must be 'none' or 'fourier', got {out_shaper!r}")
        self.n_control = n_control
        # Signal-acting columns fold into the input gain; the rest are system-acting
        # and drive the FiLM. signal_idx is clamped to valid columns.
        self.signal_idx = tuple(sorted({i for i in signal_idx if 0 <= i < n_control}))
        self.system_idx = tuple(i for i in range(n_control) if i not in self.signal_idx)
        self.n_system = len(self.system_idx)
        self.channels = channels
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.kernel = kernel
        self.cond_hidden = cond_hidden
        self.saturator = saturator
        self._adaa_order = {"clamp": 0, "adaa1": 1, "adaa2": 2}[saturator]
        self.oversample = max(int(oversample), 1)
        self.os_taps = int(os_taps)
        # Oversampling adds the polyphase FIR group delay; adaa2 adds one sample.
        os_lat = (self.os_taps - 1) // self.oversample if self.oversample > 1 else 0
        self.latency_samples = os_lat + (1 if saturator == "adaa2" else 0)
        self.stft_weight = float(stft_weight)
        self.preemph_weight = float(preemph_weight)
        self.preemph_alpha = float(preemph_alpha)
        self.preemph_order = int(preemph_order)
        self.dcblock_fc = float(dcblock_fc)
        self.rect_thr = tuple(float(t) for t in rect_thr)
        self.block_act = block_act
        self.out_shaper = out_shaper
        self.shaper_k = int(shaper_k)
        self.grad_clip = float(grad_clip)
        self.n_state = int(n_state)
        self.device = torch.device(device)
        from vguitar import AUDIO_SR

        self.net = _CIRCE3Net(
            channels,
            n_blocks,
            n_layers,
            kernel,
            self.n_system,
            cond_hidden,
            self.rect_thr,
            block_act,
            out_shaper,
            self.shaper_k,
            self.n_state,
            AUDIO_SR * self.oversample,
        ).to(self.device)
        self._stream: dict[str, Any] | None = None

    def _hparams(self) -> dict[str, Any]:
        return {
            "n_control": self.n_control,
            "signal_idx": list(self.signal_idx),
            "channels": self.channels,
            "n_blocks": self.n_blocks,
            "n_layers": self.n_layers,
            "kernel": self.kernel,
            "cond_hidden": self.cond_hidden,
            "dcblock_fc": self.dcblock_fc,
            "saturator": self.saturator,
            "stft_weight": self.stft_weight,
            "preemph_weight": self.preemph_weight,
            "preemph_alpha": self.preemph_alpha,
            "preemph_order": self.preemph_order,
            "oversample": self.oversample,
            "os_taps": self.os_taps,
            "rect_thr": list(self.rect_thr),
            "block_act": self.block_act,
            "out_shaper": self.out_shaper,
            "shaper_k": self.shaper_k,
            "grad_clip": self.grad_clip,
            "n_state": self.n_state,
        }

    def _dc_ba(self) -> tuple[np.ndarray, np.ndarray]:
        from vguitar import AUDIO_SR

        r = float(np.exp(-2.0 * np.pi * self.dcblock_fc / AUDIO_SR))
        return np.array([1.0, -1.0], dtype=np.float64), np.array([1.0, -r], dtype=np.float64)

    # --- control split helpers -------------------------------------------
    def _control_vec(self, c: np.ndarray | None) -> np.ndarray:
        if c is None:
            return np.zeros(self.n_control, dtype=np.float32)
        cv = np.asarray(c, dtype=np.float32).reshape(-1)
        if cv.shape[0] != self.n_control:
            raise ValueError(f"control has {cv.shape[0]} entries, expected {self.n_control}")
        return cv

    def _signal_gain(self, cv: np.ndarray) -> float:
        """Product of the signal-acting control values (1.0 if none)."""
        if not self.signal_idx:
            return 1.0
        return float(np.prod([cv[i] for i in self.signal_idx]))

    def _system_vec(self, cv: np.ndarray) -> np.ndarray | None:
        if self.n_system == 0:
            return None
        return np.ascontiguousarray(cv[list(self.system_idx)], dtype=np.float32)

    def _os_dataset(self, ds: Dataset) -> Dataset:
        """Upsample a dataset to the internal oversampled rate for training."""
        from vguitar.data import Dataset as _DS

        osf, taps = self.oversample, self.os_taps
        x2 = _os_resample(np.ascontiguousarray(ds.x, np.float32), osf, taps, up=True)
        y2 = _os_resample(np.ascontiguousarray(ds.y, np.float32), osf, taps, up=True)
        n = min(len(x2), len(y2))
        c2 = None
        if ds.controls is not None:
            c2 = np.repeat(np.ascontiguousarray(ds.controls, np.float32), osf, axis=0)
            n = min(n, len(c2))
            c2 = c2[:n]
        return _DS(
            x2[:n],
            y2[:n],
            ds.sr * osf,
            controls=c2,
            control_names=getattr(ds, "control_names", None),
            control_kinds=getattr(ds, "control_kinds", None),
        )

    # --- learning --------------------------------------------------------
    def fit(
        self, train: Dataset, val: Dataset | None = None, cfg: TrainConfig | None = None
    ) -> FitReport:
        """Train: signal controls fold into the input, FiLM conditions on system."""
        cfg = cfg or TrainConfig()
        if train.controls is None or train.n_controls != self.n_control:
            raise ValueError(
                f"CIRCE3 needs a dataset with {self.n_control} control column(s); "
                f"got {train.n_controls}"
            )
        torch.manual_seed(cfg.seed)
        dev = torch.device(pick_device("auto"))
        self.device = dev
        self.net.to(dev)
        # Oversampling: train the net at the internal rate on upsampled data, so it
        # learns to reproduce the (alias-free) target there; inference downsamples.
        # The receptive field / windows scale with the factor to keep memory-in-time.
        if self.oversample > 1:
            train = self._os_dataset(train)
            val = self._os_dataset(val) if val is not None else None
        seq_len = cfg.seq_len * self.oversample
        warmup = min(cfg.warmup * self.oversample, seq_len - 1)

        c_all = np.asarray(train.controls, dtype=np.float32)
        if self.n_system > 0:
            sys_all = c_all[:, list(self.system_idx)]
            self.net.c_mean.copy_(torch.from_numpy(sys_all.mean(0)).to(dev))
            self.net.c_std.copy_(torch.from_numpy(sys_all.std(0) + 1e-6).to(dev))
        self.net.out_bound.copy_(torch.tensor(1.2 * float(np.max(np.abs(train.y)) + 1e-6)).to(dev))

        xb, yb, gb, sb = self._windows(train, seq_len, hop=seq_len - warmup)
        xb, yb, gb = xb.to(dev), yb.to(dev), gb.to(dev)
        sb = sb.to(dev) if sb is not None else None
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        # Cosine LR annealing to 1% of the initial rate: the fine convergence the
        # last factor in ESR needs (a flat LR plateaus well above it).
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.01
        )
        gen = torch.Generator().manual_seed(cfg.seed)

        history: dict[str, list[float]] = {"train_loss": [], "val_esr": []}
        best_val, best_state = float("inf"), self._clone_state()
        for _ in range(cfg.epochs):
            self.net.train()
            perm = torch.randperm(xb.shape[0], generator=gen)
            ep, n_batch = 0.0, 0
            for i in range(0, len(perm), cfg.batch_size):
                sel = perm[i : i + cfg.batch_size]
                xin = xb[sel] * gb[sel][:, None]  # fold signal gain into input
                csys = sb[sel] if sb is not None else None
                pred = self.net(xin, csys)[:, warmup:]
                tgt = yb[sel][:, warmup:]
                loss = esr_loss(pred, tgt)
                if self.stft_weight:
                    loss = loss + self.stft_weight * multi_stft_loss(pred, tgt)
                if self.preemph_weight:
                    loss = loss + self.preemph_weight * preemph_esr_loss(
                        pred, tgt, self.preemph_alpha, order=self.preemph_order
                    )
                opt.zero_grad()
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                opt.step()
                ep += loss.item()
                n_batch += 1
            history["train_loss"].append(ep / max(n_batch, 1))
            ve = self._eval_esr(val, cfg) if val is not None else history["train_loss"][-1]
            history["val_esr"].append(ve)
            if ve < best_val:
                best_val, best_state = ve, self._clone_state()
            sched.step()

        self._load_state(best_state)
        to_inference_cpu(self)
        self.reset()
        return FitReport(history=history, info={"best_val_esr": best_val})

    def _clone_state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.net.state_dict().items()}

    def _load_state(self, state: dict[str, torch.Tensor]) -> None:
        self.net.load_state_dict(state)

    def _windows(
        self, ds: Dataset, seq_len: int, hop: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Tile into (x, y, signal_gain, system_control) windows (control at start)."""
        assert ds.controls is not None
        x = np.ascontiguousarray(ds.x, dtype=np.float32)
        y = np.ascontiguousarray(ds.y, dtype=np.float32)
        c = np.ascontiguousarray(ds.controls, dtype=np.float32)
        n = len(ds) - seq_len + 1
        starts = np.arange(0, max(n, 1), hop)
        idx = starts[:, None] + np.arange(seq_len)[None, :]
        xb = torch.from_numpy(x[idx])
        yb = torch.from_numpy(y[idx])
        c0 = c[starts]  # (n_win, n_control)
        gain = (
            np.prod(c0[:, list(self.signal_idx)], axis=1)
            if self.signal_idx
            else np.ones(len(starts), np.float32)
        )
        gb = torch.from_numpy(np.ascontiguousarray(gain, dtype=np.float32))
        sb = (
            torch.from_numpy(np.ascontiguousarray(c0[:, list(self.system_idx)], dtype=np.float32))
            if self.n_system > 0
            else None
        )
        return xb, yb, gb, sb

    @torch.no_grad()
    def _eval_esr(self, ds: Dataset, cfg: TrainConfig) -> float:
        """Mean per-segment validation ESR on the torch forward (train device)."""
        from vguitar.metrics import esr

        if ds.controls is None:
            return float("inf")
        self.net.eval()
        errs = []
        for s, e in _segments(ds.controls):
            cv = np.ascontiguousarray(ds.controls[s], dtype=np.float32)
            g = self._signal_gain(cv)
            xin = torch.from_numpy((ds.x[s:e] * g).astype(np.float32)).to(self.device).view(1, -1)
            sysv = self._system_vec(cv)
            csys = torch.from_numpy(sysv).to(self.device).view(1, -1) if sysv is not None else None
            pred = self.net(xin, csys).view(-1).cpu().numpy()
            w = min(cfg.warmup, (e - s) - 1)
            errs.append(esr(ds.y[s:e][w:], pred[w:]))
        return float(np.mean(errs)) if errs else float("inf")

    # --- offline inference -----------------------------------------------
    def process(self, x: np.ndarray, c: np.ndarray | None = None) -> np.ndarray:
        """Offline forward at a CONSTANT control ``c`` (bit-exact with streaming)."""
        self.net.eval()
        xv = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        self.reset()
        return self.process_block(xv, c)

    # --- streaming inference ---------------------------------------------
    def reset(self) -> None:
        self._stream = None

    def _make_adaa(self) -> Any:
        if self._adaa_order == 0:
            return None
        from vguitar.nonlinear.adaa import HARDCLIP, ADAAProcessor

        return ADAAProcessor(HARDCLIP, order=self._adaa_order)

    def _saturate(self, raw: np.ndarray, bound: float, proc: Any) -> np.ndarray:
        if proc is None:
            return np.clip(raw, -bound, bound).astype(np.float32)
        return (bound * proc.process_block(raw / bound)).astype(np.float32)

    def _build_stream(self) -> dict[str, Any]:
        c = self.channels
        sd = {k: v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}
        n_lyr = self.n_blocks * self.n_layers
        dil = [2 ** (i % self.n_layers) for i in range(n_lyr)]
        layers = [
            (
                sd[f"layers.{i}.conv.weight"],
                sd[f"layers.{i}.conv.bias"],
                sd[f"layers.{i}.res.weight"][:, :, 0],
                sd[f"layers.{i}.res.bias"],
                sd[f"layers.{i}.skip.weight"][:, :, 0],
                sd[f"layers.{i}.skip.bias"],
            )
            for i in range(n_lyr)
        ]
        mixed = self.net.block_act == "mixed"
        st: dict[str, Any] = {
            "c": c,
            "k": self.kernel,
            "dil": dil,
            "n_lyr": n_lyr,
            "block_act": self.net.block_act,
            "sizes": _mixed_sizes(c),
            "alphas": [sd[f"layers.{i}.act.alpha"] if mixed else None for i in range(n_lyr)],
            "w_in": sd["input.weight"][:, :, 0],  # (channels, n_in)
            "b_in": sd["input.bias"],
            "rect_thr": self.net.rect_thr,
            "layers": layers,
            "o1w": sd["out.1.weight"][:, :, 0],
            "o1b": sd["out.1.bias"],
            "o3w": sd["out.3.weight"][:, :, 0],
            "o3b": sd["out.3.bias"],
            "n_system": self.n_system,
            "out_bound": float(sd["out_bound"]),
            "shaper": self.out_shaper,
            "shaper_c": sd.get("shaper_c"),
            "shaper_w": sd.get("shaper_w"),
            "buf": [np.zeros((c, (self.kernel - 1) * d), dtype=np.float32) for d in dil],
            "dc_on": self.dcblock_fc > 0.0,
            "dc_ba": self._dc_ba(),
            "dc_zi": np.zeros(1, dtype=np.float64),
            "adaa": self._make_adaa(),
            "os": _OverSampler(self.oversample, self.os_taps) if self.oversample > 1 else None,
            "n_state": self.net.n_state,
        }
        if self.net.n_state > 0:
            # leaky-integrator decays a_k = sigmoid(logit); stream each as a one-pole
            # lfilter with carried state (zi) -> identical recurrence to _onepole_scan.
            st["iir_a"] = (1.0 / (1.0 + np.exp(-sd["a_logit"]))).astype(np.float64)
            st["iir_zi"] = [np.zeros(1, dtype=np.float64) for _ in range(self.net.n_state)]
        if self.n_system > 0:
            st.update(
                cw0=sd["cond.0.weight"],
                cb0=sd["cond.0.bias"],
                cw1=sd["cond.2.weight"],
                cb1=sd["cond.2.bias"],
                c_mean=sd["c_mean"],
                c_std=sd["c_std"],
            )
        return st

    def _film_np(self, s: dict[str, Any], c_sys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Numpy FiLM on the system control -> per-layer (gamma, beta), (n_lyr, C)."""
        cn = (c_sys - s["c_mean"]) / s["c_std"]
        gb = s["cw1"] @ np.tanh(s["cw0"] @ cn + s["cb0"]) + s["cb1"]
        gb = gb.reshape(s["n_lyr"], 2, s["c"])
        return (1.0 + np.tanh(gb[:, 0, :])).astype(np.float32), gb[:, 1, :].astype(np.float32)

    def process_block(self, x: np.ndarray, c: np.ndarray | None = None) -> np.ndarray:
        """Stream one block: signal gain folded into input, FiLM on system control."""
        if self._stream is None:
            self._stream = self._build_stream()
        s = self._stream
        cv = self._control_vec(c)
        g = self._signal_gain(cv)
        xb = (np.ascontiguousarray(x, dtype=np.float32).reshape(-1) * g).astype(np.float32)
        if xb.shape[0] == 0:
            return np.empty(0, dtype=np.float32)
        if s["os"] is not None:
            xb = s["os"].up(xb)  # run the network at the internal oversampled rate
        nb = xb.shape[0]
        gamma = beta = None
        if s["n_system"] > 0:
            sysv = self._system_vec(cv)
            assert sysv is not None
            gamma, beta = self._film_np(s, sysv)
        ch, k, dil = s["c"], s["k"], s["dil"]
        feats = _rect_feats_np(xb, s["rect_thr"]) if s["rect_thr"] else xb[None, :]
        if s["n_state"] > 0:
            from scipy.signal import lfilter

            state = np.empty((s["n_state"], nb), dtype=np.float32)
            for j in range(s["n_state"]):
                aj = s["iir_a"][j]
                yj, s["iir_zi"][j] = lfilter([1.0 - aj], [1.0, -aj], xb, zi=s["iir_zi"][j])
                state[j] = yj
            feats = np.concatenate([feats, state], axis=0)
        h = s["w_in"] @ feats + s["b_in"][:, None]
        skip = np.zeros((ch, nb), dtype=np.float32)
        for i, (cw, cb, rw, rb, sw, sb) in enumerate(s["layers"]):
            if gamma is not None and beta is not None:
                h = gamma[i][:, None] * h + beta[i][:, None]  # FiLM before the layer
            d = dil[i]
            ctx = np.concatenate([s["buf"][i], h], axis=1)
            conv = cb[:, None] + sum(cw[:, :, t] @ ctx[:, t * d : t * d + nb] for t in range(k))
            if k > 1:
                s["buf"][i] = ctx[:, -(k - 1) * d :].copy()
            if s["block_act"] == "mixed":
                g = _mixed_act_np(conv, s["sizes"], s["alphas"][i])
            else:
                g = np.tanh(conv[:ch]) * (1.0 / (1.0 + np.exp(-np.clip(conv[ch:], -30.0, 30.0))))
            h = h + (rw @ g + rb[:, None])
            skip += sw @ g + sb[:, None]
        o = np.maximum(skip, 0.0)
        o = np.maximum(s["o1w"] @ o + s["o1b"][:, None], 0.0)
        o = s["o3w"] @ o + s["o3b"][:, None]
        o0 = o[0]
        if s["shaper"] == "fourier":
            kk = np.arange(1, s["shaper_c"].shape[0] + 1, dtype=np.float32)
            w = float(s["shaper_w"][0])
            o0 = o0 + (s["shaper_c"][:, None] * np.sin(kk[:, None] * w * o0[None, :])).sum(0)
        a = s["out_bound"]
        out = self._saturate(o0.astype(np.float32), a, s["adaa"])  # saturate at the OS rate
        if s["os"] is not None:
            out = s["os"].down(out)  # back to the base rate (alias-free)
        if s["dc_on"]:
            from scipy.signal import lfilter

            b, av = s["dc_ba"]
            filtered, s["dc_zi"] = lfilter(b, av, out, zi=s["dc_zi"])
            out = np.clip(filtered, -a, a).astype(np.float32)
        return out

    # --- persistence -----------------------------------------------------
    def save(self, path: str | Path) -> None:
        torch.save({"hparams": self._hparams(), "state_dict": self.net.state_dict()}, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> CIRCE3:
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        hp = dict(blob["hparams"])
        hp["signal_idx"] = tuple(hp.get("signal_idx", (0,)))
        model = cls(**hp)
        model.net.load_state_dict(blob["state_dict"])
        model.reset()
        return model

    def num_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())


def _segments(controls: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous spans over which the control row is constant."""
    controls = np.asarray(controls)
    if len(controls) == 0:
        return []
    change = np.any(np.diff(controls, axis=0) != 0, axis=1)
    bounds = [0, *(np.flatnonzero(change) + 1).tolist(), len(controls)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
