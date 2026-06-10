"""Wiener-Hammerstein (WH) block-oriented emulator.

A WH model factors a nonlinear system into a **linear filter H1 -> static
nonlinearity g -> linear filter H2** sandwich. It is the canonical block-oriented
structure for guitar distortion/amp modelling: the two FIR filters carry all the
frequency-dependent (memory) behaviour while a single memoryless waveshaper does
the saturating, harmonic-generating work in between
(Eichas & Zoelzer, "Black-Box Modeling of Distortion Circuits with Block-Oriented
Models", DAFx-17; "Virtual Analog Modeling of Guitar Amplifiers with
Wiener-Hammerstein Models", DAGA-18).

Method here
-----------
* **Structure.** ``y = H2( g( H1(x) ) )`` with ``H1`` an ``mem1``-tap FIR,
  ``H2`` an ``mem2``-tap FIR, and ``g`` a degree-``poly_deg`` polynomial
  waveshaper.
* **Identification.** ``H1`` is seeded from a ridge (Tikhonov) linear least
  squares fit ``x -> y`` (the best purely-linear model is a good first guess for
  the input filter). ``H1``, the polynomial coefficients of ``g`` and ``H2`` are
  then jointly refined by a short, deterministic Adam optimisation of the
  error-to-signal ratio (ESR, the standard amp-modelling objective; Wright et
  al., DAFx-19). Training evaluates ``g`` pointwise (a plain polynomial) so the
  gradients are exact.
* **Inference anti-aliasing.** At run time the polynomial ``g`` is evaluated with
  first-order antiderivative anti-aliasing (ADAA; Parker et al., DAFx-16) instead
  of pointwise, suppressing the alias energy the waveshaper would otherwise fold
  below Nyquist. Because ``g`` is a polynomial its antiderivative is closed-form
  (term-by-term), so ADAA is exact and cheap. ``process`` and ``process_block``
  both use this path, so they agree to float32 round-off.

Accuracy
--------
WH is accurate and very cheap for *mild* distortion, where a single static
nonlinearity captures the curve well. Under heavy clipping the true circuit's
dynamic (memory-dependent) nonlinearity is no longer well approximated by one
memoryless ``g``, and the WH ESR can exceed 1 (i.e. worse than predicting
silence) -- a limitation noted by Eichas & Zoelzer (DAFx-17). For those circuits
the Volterra and neural models in this package are the better fit.

Streaming contract
-------------------
``H1``/``H2`` keep ``scipy.signal.lfilter`` states and the ADAA waveshaper keeps
its one-sample memory, so ``reset()`` + a sequence of ``process_block`` calls
reproduce a single ``process`` pass within ``~1e-5`` (well inside the default
``check_streaming`` tolerance of ``1e-4``); latency is 0.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from scipy.signal import lfilter, lfilter_zi

from vguitar.metrics import esr
from vguitar.models.base import FitReport, Model, register_model
from vguitar.nonlinear.adaa import NL, adaa1

if TYPE_CHECKING:
    from vguitar.config import TrainConfig
    from vguitar.data import Dataset


@register_model
class WienerHammerstein(Model):
    """Wiener-Hammerstein model: FIR ``H1`` -> polynomial ``g`` -> FIR ``H2``.

    Args:
        mem1: Number of taps in the input filter ``H1``.
        mem2: Number of taps in the output filter ``H2``.
        poly_deg: Degree of the static polynomial nonlinearity ``g`` (so ``g``
            has ``poly_deg + 1`` coefficients including the constant term).
    """

    name: ClassVar[str] = "wh"
    description: ClassVar[str] = "Wiener-Hammerstein: FIR -> polynomial waveshaper -> FIR"
    latency_samples: int = 0

    def __init__(self, mem1: int = 64, mem2: int = 64, poly_deg: int = 7) -> None:
        if min(mem1, mem2) < 1 or poly_deg < 1:
            raise ValueError("mem1, mem2 must be >= 1 and poly_deg >= 1")
        self.mem1 = int(mem1)
        self.mem2 = int(mem2)
        self.poly_deg = int(poly_deg)

        # Learned parameters. Untrained defaults form an identity-ish system
        # (H1 = unit impulse, g(u) = u, H2 = unit impulse) so an un-fit instance
        # is well defined and load() can restore over it.
        self.h1 = np.zeros(self.mem1, dtype=np.float64)
        self.h1[0] = 1.0
        self.h2 = np.zeros(self.mem2, dtype=np.float64)
        self.h2[0] = 1.0
        # Polynomial coeffs in increasing-power order: g(u) = sum c_k u**k.
        self.coef = np.zeros(self.poly_deg + 1, dtype=np.float64)
        self.coef[1] = 1.0  # identity slope

        self._nl = self._make_nl()
        self.reset()

    # --- the polynomial nonlinearity, as an ADAA-ready NL -----------------
    def _make_nl(self) -> NL:
        """Bundle ``g`` and its first antiderivative ``F1`` for first-order ADAA.

        ``g(u) = sum_k c_k u**k`` is a polynomial, so ``F1(u) = sum_k
        c_k/(k+1) u**(k+1)`` is exact and closed-form. ``np.polyval`` wants
        coefficients in *decreasing* power order, hence the ``[::-1]``. ``F2`` is
        unused by first-order ADAA but :class:`NL` requires it, so we supply the
        analytic second antiderivative too (cheap, keeps the object valid).
        """
        c = self.coef
        k = np.arange(c.size)
        f_coef = c[::-1]  # highest power first, for polyval
        f1_coef = np.concatenate(([0.0], c / (k + 1.0)))[::-1]
        f2_coef = np.concatenate(([0.0, 0.0], c / ((k + 1.0) * (k + 2.0))))[::-1]
        return NL(
            name="poly",
            f=lambda u: np.polyval(f_coef, u),
            F1=lambda u: np.polyval(f1_coef, u),
            F2=lambda u: np.polyval(f2_coef, u),
        )

    # --- identification ---------------------------------------------------
    def fit(
        self, train: Dataset, val: Dataset | None = None, cfg: TrainConfig | None = None
    ) -> FitReport:
        """Seed ``H1`` from a ridge linear fit, then jointly refine via Adam-ESR.

        The linear seed (best FIR ``x -> y``) gives the optimiser a sensible
        input filter; ``g`` starts as identity and ``H2`` as a unit impulse, so
        the initial system is the linear model itself and Adam only has to bend
        in the nonlinearity. Optimisation is deterministic (fixed seed, full-batch
        truncated windows) and uses a pointwise polynomial for exact gradients.
        """
        x = np.asarray(train.x, dtype=np.float64)
        y = np.asarray(train.y, dtype=np.float64)
        seed = cfg.seed if cfg is not None else 0
        lr = cfg.lr if cfg is not None else 5e-3
        steps = 400

        self.h1 = _ridge_fir(x, y, self.mem1)
        self.h1 /= np.linalg.norm(self.h1) + 1e-12  # fix gain ambiguity into g/H2

        history = self._refine(x, y, seed=seed, lr=lr, steps=steps)
        self._nl = self._make_nl()
        self.reset()

        info: dict[str, float] = {"mem1": self.mem1, "mem2": self.mem2, "poly_deg": self.poly_deg}
        if val is not None and len(val) > 0:
            info["val_esr"] = esr(val.y, self.process(val.x))
        return FitReport(history=history, info=info)

    def _refine(
        self, x: np.ndarray, y: np.ndarray, *, seed: int, lr: float, steps: int
    ) -> dict[str, list[float]]:
        """Joint Adam refinement of ``H1``, ``coef`` and ``H2`` on ESR (DAFx-19).

        Implemented in torch with FIR as 1-D convolution and ``g`` as a Horner
        polynomial — all differentiable — so a few hundred Adam steps tune every
        block at once. Pointwise (non-ADAA) ``g`` is used for training; ADAA is an
        inference-only refinement of the same coefficients.
        """
        torch.manual_seed(seed)
        xt = torch.from_numpy(x).to(torch.float64)
        yt = torch.from_numpy(y).to(torch.float64)

        h1 = torch.tensor(self.h1, dtype=torch.float64, requires_grad=True)
        h2 = torch.tensor(self.h2, dtype=torch.float64, requires_grad=True)
        coef = torch.tensor(self.coef, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([h1, h2, coef], lr=lr)

        eps = 1e-12
        y_energy = float(torch.sum(yt * yt)) + eps
        hist: list[float] = []
        for _ in range(steps):
            opt.zero_grad()
            u = _fir(xt, h1)
            v = _poly(u, coef)
            yhat = _fir(v, h2)
            loss = torch.sum((yt - yhat) ** 2) / y_energy
            loss.backward()
            opt.step()
            hist.append(loss.item())

        self.h1 = h1.detach().numpy().astype(np.float64)
        self.h2 = h2.detach().numpy().astype(np.float64)
        self.coef = coef.detach().numpy().astype(np.float64)
        return {"train_esr": hist}

    # --- offline inference ------------------------------------------------
    def process(self, x: np.ndarray) -> np.ndarray:
        """Whole-signal ``H2(ADAA(g(H1(x))))``; returns float32, same length."""
        xf = np.asarray(x, dtype=np.float64).reshape(-1)
        u = lfilter(self.h1, [1.0], xf)
        v, _ = adaa1(self._nl, u, 0.0)
        y = lfilter(self.h2, [1.0], v)
        return y.astype(np.float32)

    # --- streaming inference ----------------------------------------------
    def reset(self) -> None:
        """Clear both FIR delay lines and the ADAA one-sample memory."""
        # lfilter_zi gives the steady-state init for a step; scaling by 0 yields
        # the zero-state filter memory of the right length for each FIR.
        self._zi1 = lfilter_zi(self.h1, [1.0]) * 0.0
        self._zi2 = lfilter_zi(self.h2, [1.0]) * 0.0
        self._x_prev = 0.0

    def process_block(self, x: np.ndarray) -> np.ndarray:
        """Process one block, advancing all three states; float32, same length."""
        xf = np.asarray(x, dtype=np.float64).reshape(-1)
        u, self._zi1 = lfilter(self.h1, [1.0], xf, zi=self._zi1)
        v, self._x_prev = adaa1(self._nl, u, self._x_prev)
        y, self._zi2 = lfilter(self.h2, [1.0], v, zi=self._zi2)
        return y.astype(np.float32)

    # --- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Persist hyperparameters and learned arrays to a ``.npz`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:  # file handle => no '.npz' appended; save/load symmetric
            np.savez(
                f,
                mem1=np.int64(self.mem1),
                mem2=np.int64(self.mem2),
                poly_deg=np.int64(self.poly_deg),
                h1=self.h1,
                h2=self.h2,
                coef=self.coef,
            )

    @classmethod
    def load(cls, path: str | Path) -> WienerHammerstein:
        """Reconstruct an instance and restore its learned parameters."""
        with np.load(path, allow_pickle=False) as f:
            model = cls(mem1=int(f["mem1"]), mem2=int(f["mem2"]), poly_deg=int(f["poly_deg"]))
            model.h1 = np.asarray(f["h1"], dtype=np.float64)
            model.h2 = np.asarray(f["h2"], dtype=np.float64)
            model.coef = np.asarray(f["coef"], dtype=np.float64)
        model._nl = model._make_nl()
        model.reset()
        return model

    # --- introspection ----------------------------------------------------
    def num_params(self) -> int:
        """Count of learned parameters: ``mem1 + mem2 + poly_deg + 1``."""
        return self.mem1 + self.mem2 + (self.poly_deg + 1)


# --- helpers --------------------------------------------------------------
def _ridge_fir(x: np.ndarray, y: np.ndarray, taps: int, lam: float = 1e-6) -> np.ndarray:
    """Ridge (Tikhonov) least-squares FIR fit ``x -> y`` with ``taps`` lags.

    Solves ``min_h ||X h - y||^2 + lam ||h||^2`` on the normal equations, where
    ``X`` is the causal Toeplitz design matrix of lagged inputs. The small ``lam``
    keeps the solve well conditioned when the input is band-limited. This best
    purely-linear filter is the seed for ``H1`` (Eichas & Zoelzer, DAFx-17).
    """
    n = x.size
    cols = [np.concatenate((np.zeros(k), x[: n - k])) for k in range(taps)]
    design = np.stack(cols, axis=1)  # (n, taps)
    gram = design.T @ design + lam * np.eye(taps)
    return np.linalg.solve(gram, design.T @ y)


def _fir(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Causal FIR ``y[n] = sum_k h[k] x[n-k]`` via padded 1-D convolution.

    Pads ``len(h)-1`` zeros on the left so the output is causal and the same
    length as ``x``. ``conv1d`` correlates, so ``h`` is flipped to realise a true
    convolution.
    """
    taps = h.shape[0]
    xp = torch.nn.functional.pad(x.view(1, 1, -1), (taps - 1, 0))
    w = h.flip(0).view(1, 1, -1)
    return torch.nn.functional.conv1d(xp, w).view(-1)


def _poly(u: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    """Evaluate ``sum_k coef[k] u**k`` (increasing power) via Horner's scheme."""
    y = torch.zeros_like(u)
    for c in reversed(coef):
        y = y * u + c
    return y
