"""Recurrent (LSTM/GRU) real-time amplifier emulator.

A single recurrent layer mapping the scalar input sample to a ``hidden``-wide
state, followed by an output ``Linear(hidden, 1)`` and a residual skip
``y = linear(rnn(x)) + x``. This is the architecture of Wright et al.,
"Real-Time Guitar Amplifier Emulation with Deep Learning" (Appl. Sci. 2020) and
its DAFx-19 precursor "Real-Time Black-Box Modelling with Recurrent Neural
Networks": a tiny stateful network that learns the amp's input/output map
directly. The skip connection lets the recurrent core model only the *deviation*
from the dry signal, which they found speeds convergence and lowers ESR markedly.

Why RNNs for real time: an LSTM/GRU parallelizes poorly (each step depends on the
previous), so it cannot exploit wide-block vectorization the way a TCN/SSM can.
But its per-step cost is small and *constant*, so it hits real time at **any**
block size, down to a single sample — making it the safe, latency-robust real-time
choice (Comunita, Steinmetz & Reiss, "Differentiable All-Pole Filters..."/RNN
benchmarking, 2025). It is the natural baseline for this benchmark.

Training is truncated backpropagation through time (TBPTT): the signal is cut into
``cfg.seq_len`` windows, the hidden state is carried *but detached* between
windows, and the first ``cfg.warmup`` samples of every window are excluded from the
loss so the loss is never charged for cold-start transients (Wright et al.).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from torch import nn

from vguitar.config import TrainConfig
from vguitar.losses import combined_loss, esr_loss
from vguitar.models.base import FitReport, Model, register_model

if TYPE_CHECKING:
    from vguitar.data import Dataset

_State = tuple[torch.Tensor, torch.Tensor] | torch.Tensor


class _RNNNet(nn.Module):
    """One recurrent layer (input size 1) + ``Linear`` head with a residual skip.

    ``forward`` runs a ``(B, T, 1)`` batch and returns ``(out, state)`` so callers
    can carry/detach the hidden state for truncated BPTT and streaming.
    """

    def __init__(self, hidden: int, cell: str) -> None:
        super().__init__()
        cell = cell.lower()
        if cell == "lstm":
            self.rnn: nn.RNNBase = nn.LSTM(1, hidden, batch_first=True)
        elif cell == "gru":
            self.rnn = nn.GRU(1, hidden, batch_first=True)
        else:
            raise ValueError(f"cell must be 'lstm' or 'gru', got {cell!r}")
        self.cell = cell
        self.hidden = hidden
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, state: _State | None = None) -> tuple[torch.Tensor, _State]:
        z, new_state = self.rnn(x, state)
        return self.head(z) + x, new_state  # residual skip (Wright et al.)


@register_model
class RNN(Model):
    """LSTM/GRU amp emulator; see module docstring for the method and citations."""

    name: ClassVar[str] = "rnn"
    description: ClassVar[str] = "Recurrent (LSTM/GRU) amp model with residual skip (Wright 2020)."
    #: process() uses zero init state, matching reset()+process_block() exactly.
    latency_samples: int = 0

    def __init__(self, hidden: int = 32, cell: str = "lstm", device: str = "cpu") -> None:
        self.hidden = int(hidden)
        self.cell = cell.lower()
        self.device = torch.device(device)
        self.net = _RNNNet(self.hidden, self.cell).to(self.device)
        self._state: _State | None = None  # streaming hidden state

    # --- learning ---------------------------------------------------------
    def fit(
        self, train: Dataset, val: Dataset | None = None, cfg: TrainConfig | None = None
    ) -> FitReport:
        """Truncated-BPTT training with ESR(+multi-STFT) loss; keeps best-val weights.

        The training signal is split into non-overlapping ``seq_len`` windows
        forming one batch; the hidden state flows window-to-window but is detached
        each step (TBPTT), and the first ``warmup`` samples are dropped from the
        loss so warm-up transients are not penalized (Wright et al., DAFx-19).
        """
        cfg = cfg or TrainConfig()
        torch.manual_seed(cfg.seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        xb, yb = self._windows(train, cfg)  # (B, W, 1) each
        vx, vy = self._whole(val) if val is not None else (None, None)

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_state: dict[str, torch.Tensor] | None = None

        for _ in range(cfg.epochs):
            self.net.train()
            opt.zero_grad()
            loss = self._tbptt_loss(xb, yb, cfg)
            loss.backward()
            opt.step()
            history["train_loss"].append(float(loss.detach()))

            vloss = self._val_loss(vx, vy)
            if vloss is not None:
                history["val_loss"].append(vloss)
                if vloss < best_val:
                    best_val = vloss
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}

        if best_state is not None:
            self.net.load_state_dict(best_state)
        if not history["val_loss"]:
            history.pop("val_loss")
        return FitReport(history=history, info={"cell": self.cell, "hidden": self.hidden})

    def _tbptt_loss(self, xb: torch.Tensor, yb: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
        """One forward/loss pass over chunked windows, detaching state between chunks."""
        n = xb.shape[1]
        chunk = max(1, cfg.seq_len)
        warm = min(cfg.warmup, chunk - 1)
        state: _State | None = None
        preds: list[torch.Tensor] = []
        targs: list[torch.Tensor] = []
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            out, state = self.net(xb[:, s:e], state)
            state = _detach_state(state)
            preds.append(out[:, warm:, 0])
            targs.append(yb[:, s:e, 0][:, warm:])
        y_pred = torch.cat(preds, dim=1)
        y_true = torch.cat(targs, dim=1)
        return combined_loss(y_pred, y_true, stft_weight=0.5)

    def _val_loss(self, vx: torch.Tensor | None, vy: torch.Tensor | None) -> float | None:
        if vx is None or vy is None:
            return None
        self.net.eval()
        with torch.no_grad():
            out, _ = self.net(vx)
            return float(esr_loss(out[:, :, 0], vy[:, :, 0]))

    # --- offline inference ------------------------------------------------
    def process(self, x: np.ndarray) -> np.ndarray:
        """Run the whole sequence from a zero initial state (matches streaming)."""
        xt = self._to_tensor(x)  # (1, T, 1)
        self.net.eval()
        with torch.no_grad():
            out, _ = self.net(xt)
        return out[0, :, 0].cpu().numpy().astype(np.float32)

    # --- streaming inference ----------------------------------------------
    def reset(self) -> None:
        """Zero the carried hidden state so the next block starts cold."""
        self._state = None

    def process_block(self, x: np.ndarray) -> np.ndarray:
        """Process one block, carrying ``(h, c)``/``h`` across calls (latency 0)."""
        xt = self._to_tensor(x)
        self.net.eval()
        with torch.no_grad():
            out, self._state = self.net(xt, self._state)
            self._state = _detach_state(self._state)
        return out[0, :, 0].cpu().numpy().astype(np.float32)

    # --- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save hyperparameters + ``state_dict`` so ``load`` fully reconstructs."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob: dict[str, Any] = {
            "hyper": {"hidden": self.hidden, "cell": self.cell},
            "state_dict": self.net.state_dict(),
        }
        torch.save(blob, path)

    @classmethod
    def load(cls, path: str | Path) -> RNN:
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        model = cls(**blob["hyper"])
        model.net.load_state_dict(blob["state_dict"])
        return model

    # --- introspection ----------------------------------------------------
    def num_params(self) -> int:
        """Total number of learned parameters (recurrent weights + linear head)."""
        return sum(p.numel() for p in self.net.parameters())

    # --- helpers ----------------------------------------------------------
    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        """``(N,)`` numpy -> ``(1, N, 1)`` float32 tensor on the model device."""
        a = np.asarray(x, dtype=np.float32).reshape(-1)
        return torch.from_numpy(a).to(self.device).view(1, -1, 1)

    def _whole(self, ds: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
        """A dataset as a single ``(1, T, 1)`` (input, target) pair."""
        return self._to_tensor(ds.x), self._to_tensor(ds.y)

    def _windows(self, ds: Dataset, cfg: TrainConfig) -> tuple[torch.Tensor, torch.Tensor]:
        """Cut the training signal into ``batch_size`` equal-length ``(B, W, 1)`` windows.

        Splitting into a batch of parallel sub-sequences is the standard way to
        train an RNN on one long recording: it both speeds training and lets the
        TBPTT state flow independently per stream (Wright et al.).
        """
        x = np.asarray(ds.x, dtype=np.float32).reshape(-1)
        y = np.asarray(ds.y, dtype=np.float32).reshape(-1)
        n_seq = max(1, cfg.batch_size)
        w = len(x) // n_seq
        if w < 2:  # too little data to batch: fall back to one full sequence
            n_seq, w = 1, len(x)
        x = x[: n_seq * w].reshape(n_seq, w, 1)
        y = y[: n_seq * w].reshape(n_seq, w, 1)
        xt = torch.from_numpy(x).to(self.device)
        yt = torch.from_numpy(y).to(self.device)
        return xt, yt


def _detach_state(state: _State) -> _State:
    """Detach hidden state from the graph (TBPTT / streaming) for LSTM or GRU."""
    if isinstance(state, tuple):
        return (state[0].detach(), state[1].detach())  # LSTM (h, c)
    return state.detach()  # GRU h
