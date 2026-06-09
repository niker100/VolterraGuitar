"""Central configuration.

Small frozen dataclasses, no magic. The canonical audio rate lives in
``vguitar.AUDIO_SR``; everything here defaults to it so the whole pipeline
(simulate -> dataset -> train -> benchmark -> live) shares one time base. This
was a root-cause bug in v1 (11025 Hz export vs ~400 kHz sim step vs 44.1 kHz
inference); keeping a single ``sr`` everywhere is the fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vguitar import AUDIO_SR


@dataclass(frozen=True)
class SimConfig:
    """ngspice transient-simulation settings.

    The raw simulator output (irregular adaptive timestep) is resampled onto a
    uniform ``sim_sr`` grid that is high enough to contain the circuit's
    nonlinear harmonics, then anti-alias-decimated to the dataset ``sr``. That
    decimation is what makes the *target* band-limited and alias-free.
    """

    sim_sr: int = 8 * AUDIO_SR  # 352.8 kHz uniform grid for raw SPICE output
    max_step_s: float = 1.0 / (16 * AUDIO_SR)  # ngspice .tran max step (~1.4 us)
    temperature_c: float = 27.0
    abstol: float = 1e-9
    reltol: float = 1e-4


@dataclass(frozen=True)
class DataConfig:
    """Excitation + dataset-generation settings."""

    sr: int = AUDIO_SR
    duration_s: float = 30.0
    seed: int = 0
    # Peak input amplitudes (volts) the excitation should sweep through. Broad
    # amplitude coverage is essential: Volterra/neural models diverge outside
    # the amplitude range they were trained on (a v1 failure mode).
    drive_levels: tuple[float, ...] = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
    # Fraction of conditioned-training segments rendered from a real guitar-DI
    # window instead of synthetic excitation. Putting real playing in the
    # training set closes the synthetic-vs-real generalization gap (used by the
    # CIRCE control-sweep generation; the unconditioned make_dataset ignores it).
    di_mix: float = 0.34


@dataclass(frozen=True)
class TrainConfig:
    """Training / identification settings shared by all model classes."""

    sr: int = AUDIO_SR
    seq_len: int = 4096  # truncated-BPTT / window length for sequence models
    warmup: int = 512  # samples discarded from each window's loss (state warm-up)
    batch_size: int = 40
    epochs: int = 60
    lr: float = 5e-3
    weight_decay: float = 0.0
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    device: str = "cpu"  # CPU is the realtime target; keep training honest
    seed: int = 0
    amp: bool = False  # bf16 autocast on CUDA (training-only; fp32 weights + numpy twin)


@dataclass(frozen=True)
class RealtimeConfig:
    """Live-audio engine settings."""

    sr: int = AUDIO_SR
    block_size: int = 128  # PortAudio frames per callback (~2.9 ms at 44.1 kHz)
    oversample: int = 2  # inference-time oversampling for anti-aliasing
    input_device: int | None = None
    output_device: int | None = None


@dataclass(frozen=True)
class Paths:
    data: Path = Path("data")
    outputs: Path = Path("outputs")
    runs: Path = Path("runs")
    assets: Path = Path("assets")

    def ensure(self) -> None:
        for p in (self.data, self.outputs, self.runs):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    """Top-level bundle passed around the pipeline."""

    sim: SimConfig = field(default_factory=SimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
    paths: Paths = field(default_factory=Paths)
