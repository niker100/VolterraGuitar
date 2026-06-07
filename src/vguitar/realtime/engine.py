"""Live audio engine, real-time-factor measurement, and an offline renderer.

REALTIME CONSTRAINTS (why this module is written the way it is)
--------------------------------------------------------------
The PortAudio callback runs on a high-priority audio thread and must return a
block of exactly ``block_size`` frames within one block period (~2.9 ms at
44.1 kHz / 128 frames). Anything that blocks — memory allocation, the GIL under
contention, file/console I/O, exceptions — risks an underrun (an audible click).
So the callback here:

* allocates nothing per call that it can preallocate (scratch buffers and the
  stateful resamplers are created once in :meth:`LiveEngine.start`);
* never raises: dtype/shape surprises are coerced, and any model error is
  caught and replaced with silence rather than killing the stream;
* keeps work O(block) and branch-light.

Oversampling is *opt-in* (``cfg.oversample``). Many nonlinear emulators alias
when run at the base rate; running the model at ``oversample x`` and band-limit
decimating around it pushes the alias products above Nyquist before they fold
back (the standard antialiasing-by-oversampling argument, e.g. Parker et al.,
"Reducing the Aliasing of Nonlinear Waveshaping Using Continuous-Time
Convolution", DAFx-16). We use soxr's *stateful* ``ResampleStream`` so the
anti-alias filter state is continuous across blocks (a fresh oneshot per block
would inject boundary discontinuities). Because polyphase resampling emits a
variable number of frames per chunk, a small FIFO re-blocks the model output
back to an exact ``block_size`` for PortAudio.

RTF measurement and the offline renderer need no audio hardware and are the way
to evaluate models in the benchmark / on CI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from vguitar.config import RealtimeConfig
    from vguitar.models.base import Model


def _control_for_block(control: Any, idx: int, n_blocks: int) -> np.ndarray | None:
    """A control source for conditioned playback is a constant vector ``(K,)``, a
    per-block schedule ``(n_blocks, K)``, or a callable ``block_idx -> (K,)``."""
    """Resolve the control vector for output block ``idx`` (see :data:`Control`)."""
    if control is None:
        return None
    if callable(control):
        return np.asarray(control(idx), dtype=np.float32).reshape(-1)
    arr = np.asarray(control, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return arr[min(idx, arr.shape[0] - 1)]


def measure_rtf(
    model: Model, sr: int = 44_100, block: int = 128, dur_s: float = 5.0
) -> dict[str, float | int | bool]:
    """Time ``model.process_block`` and report the real-time factor (RTF).

    Feeds consecutive random blocks of ``block`` samples through the streaming
    path for roughly ``dur_s`` seconds of audio, timing everything *after* one
    discarded warm-up block (so JIT/cache/allocation costs of the first call do
    not pollute the steady-state number).

    The RTF is ``audio_seconds / compute_seconds``: ``> 1`` means the model
    keeps up with realtime (produces audio faster than it plays). This is the
    headline efficiency metric for the benchmark.

    Returns a dict with ``rtf``, ``ms_per_block``, ``blocks`` (timed), and
    ``realtime`` (``rtf > 1.0``).
    """
    from time import perf_counter

    rng = np.random.default_rng(0)
    n_blocks = max(1, round(dur_s * sr / block))
    # Preallocate all excitation blocks so RNG cost is excluded from timing.
    blocks = [(rng.standard_normal(block).astype(np.float32) * 0.3) for _ in range(n_blocks + 1)]

    model.reset()
    model.process_block(blocks[0])  # warm-up: discarded from timing

    t0 = perf_counter()
    for b in blocks[1:]:
        model.process_block(b)
    elapsed = perf_counter() - t0

    audio_s = (n_blocks * block) / sr
    rtf = audio_s / elapsed if elapsed > 0 else float("inf")
    return {
        "rtf": rtf,
        "ms_per_block": 1e3 * elapsed / n_blocks,
        "blocks": n_blocks,
        "realtime": rtf > 1.0,
    }


def render_file(model: Model, in_path: str, out_path: str, sr: int = 44_100,
                *, control: Any = None) -> None:
    """Offline fallback: stream a wav through the model and write the result.

    Reads ``in_path`` (mono; if multichannel, the first channel is used),
    resamples to ``sr`` if needed, runs the model block-by-block through the
    *streaming* path (so the output matches what the live engine would produce),
    and writes ``out_path``. Useful for listening on machines with no audio
    device or where opening a duplex stream is impractical (CI, containers).

    ``control`` drives a conditioned model: a constant vector ``(K,)``, a per-
    block schedule ``(n_blocks, K)``, or a callable ``block_idx -> (K,)`` for a
    knob automation. ``None`` (the default) runs an unconditioned model unchanged.
    """
    import soundfile as sf
    import soxr

    x, file_sr = sf.read(in_path, dtype="float32", always_2d=True)
    x = np.ascontiguousarray(x[:, 0], dtype=np.float32)  # take first channel
    if file_sr != sr:
        x = soxr.resample(x, file_sr, sr, quality="VHQ").astype(np.float32)

    model.reset()
    m: Any = model  # conditioned models accept the extra control arg
    block = 1024
    out = np.empty_like(x)
    n_blocks = (len(x) + block - 1) // block
    for bi, i in enumerate(range(0, len(x), block)):
        chunk = x[i : i + block]
        c = _control_for_block(control, bi, n_blocks)
        y = m.process_block(chunk, c) if c is not None else m.process_block(chunk)
        out[i : i + len(chunk)] = np.asarray(y, dtype=np.float32)
    np.clip(out, -1.0, 1.0, out=out)
    sf.write(out_path, out, sr)


class LiveEngine:
    """Duplex live-audio host running a model in the PortAudio callback.

    Opens a mono-in/mono-out :class:`sounddevice.Stream` at ``cfg.sr`` with
    ``cfg.block_size`` frames per callback. With ``cfg.oversample > 1`` the
    model runs at the oversampled rate using stateful soxr resamplers (see the
    module docstring on antialiasing). Output is clamped to ``[-1, 1]``.

    Use :meth:`start` / :meth:`stop`; the engine is also a context manager.
    """

    def __init__(self, model: Model, cfg: RealtimeConfig,
                 control_fn: Callable[[int], np.ndarray] | None = None) -> None:
        self.model = model
        self.cfg = cfg
        #: Optional live control source for a conditioned model: called once per
        #: callback with the block index, returns the control vector ``(K,)``
        #: (e.g. a closure over a MIDI/OSC/GUI knob). ``None`` => unconditioned.
        self._control_fn = control_fn
        self._cbuf = np.zeros(int(getattr(model, "n_control", 0)), dtype=np.float32)
        self._block_idx = 0
        # Process function chosen once in start() to keep the callback branch-light.
        self._proc: Callable[[np.ndarray], np.ndarray] = model.process_block
        self._stream = None  # type: ignore[var-annotated]  # sounddevice.Stream, lazy import
        self._up = None  # soxr.ResampleStream sr -> os_sr
        self._down = None  # soxr.ResampleStream os_sr -> sr
        self._fifo = np.empty(0, dtype=np.float32)  # re-blocking buffer (oversampled path)

    def _proc_conditioned(self, x: np.ndarray) -> np.ndarray:
        """Fill the preallocated control buffer from ``control_fn`` and run the model."""
        cf = self._control_fn
        assert cf is not None
        vec = np.asarray(cf(self._block_idx), dtype=np.float32).reshape(-1)
        self._cbuf[:] = vec[: self._cbuf.shape[0]]
        self._block_idx += 1
        model: Any = self.model  # conditioned models accept the extra control arg
        return model.process_block(x, self._cbuf)

    # --- lifecycle --------------------------------------------------------
    def start(self) -> None:
        """Open and start the audio stream; print device + latency info."""
        import sounddevice as sd

        cfg = self.cfg
        self.model.reset()
        self._fifo = np.empty(0, dtype=np.float32)
        self._block_idx = 0
        self._proc = self._proc_conditioned if self._control_fn is not None else self.model.process_block

        os_factor = max(1, int(cfg.oversample))
        if os_factor > 1:
            import soxr

            os_sr = cfg.sr * os_factor
            # Stateful resamplers keep the anti-alias filter continuous across
            # callbacks; 'HQ' is a good latency/quality trade for live use.
            self._up = soxr.ResampleStream(cfg.sr, os_sr, 1, dtype="float32", quality="HQ")
            self._down = soxr.ResampleStream(os_sr, cfg.sr, 1, dtype="float32", quality="HQ")
            callback = self._callback_os
        else:
            self._up = self._down = None
            callback = self._callback_base

        self._stream = sd.Stream(
            samplerate=cfg.sr,
            blocksize=cfg.block_size,
            channels=1,
            dtype="float32",
            device=(cfg.input_device, cfg.output_device),
            callback=callback,
        )
        self._stream.start()
        lat_in, lat_out = self._stream.latency
        print(
            f"LiveEngine: model={self.model.name!r} sr={cfg.sr} block={cfg.block_size} "
            f"oversample={os_factor}x"
        )
        print(
            f"  in={sd.query_devices(self._stream.device[0])['name']!r} "
            f"out={sd.query_devices(self._stream.device[1])['name']!r}"
        )
        print(f"  latency: in={1e3 * lat_in:.1f} ms out={1e3 * lat_out:.1f} ms")

    def stop(self) -> None:
        """Stop and close the stream (idempotent)."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> LiveEngine:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # --- callbacks (audio thread; must not raise) -------------------------
    def _callback_base(self, indata, outdata, frames, time_info, status) -> None:
        """Base-rate path: model runs directly at ``cfg.sr``."""
        try:
            x = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
            y = np.asarray(self._proc(x), dtype=np.float32)
            if y.shape[0] != frames:  # guard: model must return same length
                y = np.resize(y, frames)
            np.clip(y, -1.0, 1.0, out=y)
            outdata[:, 0] = y
        except Exception:  # never kill the stream on a model/dtype error
            outdata.fill(0.0)

    def _callback_os(self, indata, outdata, frames, time_info, status) -> None:
        """Oversampled path: upsample -> model -> downsample, re-blocked via FIFO."""
        up_stream, down_stream = self._up, self._down
        if up_stream is None or down_stream is None:  # only set when oversample > 1
            outdata.fill(0.0)
            return
        try:
            x = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
            up = up_stream.resample_chunk(x)
            if up.size:
                proc = np.asarray(self._proc(up), dtype=np.float32)
                down = down_stream.resample_chunk(proc)
                if down.size:
                    self._fifo = np.concatenate((self._fifo, down))
            # Emit exactly ``frames``; pad with silence if the resampler is still
            # priming (the small startup delay shows up only in the first blocks).
            if self._fifo.shape[0] >= frames:
                y, self._fifo = self._fifo[:frames], self._fifo[frames:]
            else:
                y = np.zeros(frames, dtype=np.float32)
                y[: self._fifo.shape[0]] = self._fifo
                self._fifo = np.empty(0, dtype=np.float32)
            np.clip(y, -1.0, 1.0, out=y)
            outdata[:, 0] = y
        except Exception:
            outdata.fill(0.0)
