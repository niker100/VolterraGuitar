"""Tests for the torch training losses (esr / pre-emphasis / multi-STFT)."""

from __future__ import annotations

import torch

from vguitar.models.archive.losses import esr_loss, preemph_esr_loss


def test_esr_zero_on_match() -> None:
    y = torch.sin(torch.linspace(0, 20, 512))
    assert float(esr_loss(y, y)) < 1e-12


def test_preemph_zero_on_match() -> None:
    y = torch.sin(torch.linspace(0, 20, 512))
    assert float(preemph_esr_loss(y, y)) < 1e-12


def test_preemph_accepts_1d_and_batch() -> None:
    y = torch.randn(4, 512)
    yp = y + 0.01 * torch.randn(4, 512)
    assert float(preemph_esr_loss(yp, y)) >= 0.0
    # (T,) is promoted to (1, T) internally — no shape error.
    assert float(preemph_esr_loss(yp[0], y[0])) >= 0.0


def test_preemph_weights_high_frequencies_more() -> None:
    """The defining property: a high-frequency error of a given energy costs MORE
    under pre-emphasis than a low-frequency error of the SAME energy (plain ESR
    treats them equally; pre-emphasis lifts the highs into the gradient)."""
    n = 1024
    t = torch.linspace(0, 1, n)
    target = torch.sin(2 * torch.pi * 5 * t)  # low-frequency reference signal
    amp = 0.05
    lf_err = amp * torch.sin(2 * torch.pi * 7 * t)  # low-frequency error
    hf_err = amp * torch.sin(2 * torch.pi * 400 * t)  # high-frequency error, equal energy

    # Equal-energy errors => equal plain ESR ...
    esr_lf = float(esr_loss(target + lf_err, target))
    esr_hf = float(esr_loss(target + hf_err, target))
    assert abs(esr_lf - esr_hf) < 0.05 * esr_lf

    # ... but the high-frequency error costs far more under pre-emphasis.
    pe_lf = float(preemph_esr_loss(target + lf_err, target))
    pe_hf = float(preemph_esr_loss(target + hf_err, target))
    assert pe_hf > 5.0 * pe_lf
