"""Drive an arbitrary signal through a :class:`Circuit` with ngspice.

This is the data-generation heart of the project: it produces the ground-truth
``(x, y)`` pairs that every model is trained and judged against. Getting it
right (alias-free, correctly time-aligned, convergent) is what gated v1, so the
mechanism is documented in detail below.

Mechanism (the tricky part)
---------------------------
ngspice's transient solver uses an *adaptive* timestep, so we cannot just hand
it a sampled array — it asks for the source value at arbitrary times. PySpice's
``NgSpiceShared`` exposes ngspice's shared-library callback ``get_vsrc_data``
for exactly this: if a voltage source is declared with the ``external`` keyword,
ngspice calls back into Python every time it needs that source's value, passing
the current solver ``time``. We subclass ``NgSpiceShared`` and override the
callback to return the input signal sampled at ``time``.

To keep that lookup cheap *and* accurate we pre-upsample the input once to the
high uniform ``sim_sr`` grid (``vguitar.resample.resample``) and, in the
callback, do a nearest-sample read ``idx = round(time * sim_sr)``. ``sim_sr``
(default 8x audio = 352.8 kHz) is far above the solver's max step
(~1.4 us), so nearest-neighbour adds negligible error while avoiding any
per-call interpolation. (Verified empirically: the external source declared as
``Vin in 0 dc 0 external`` causes the callback to fire with ``node == "vin"``.)

We do **not** accumulate output in the ``send_data`` per-step callback (it is
called once per accepted step and is easy to get out of sync with rejected
steps). Instead we let the run complete and read the finished plot's ``time``
and ``out`` vectors directly via ``plot()`` — the single most reliable source of
truth. The raw, non-uniformly-sampled ``(t_out, y_out)`` is then resampled onto
the audio grid by ``vguitar.resample.to_audio`` (which band-limits and decimates
sim_sr -> sr, making the target alias-free; see SimConfig docstring).

Windows note: PySpice appends the ``ngspice_id`` to the DLL name
(``ngspice<id>.dll``). Only ``ngspice.dll`` ships, so we must use
``ngspice_id=0``.
"""

from __future__ import annotations

import contextlib
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from vguitar.config import SimConfig

if TYPE_CHECKING:
    from vguitar.circuits.base import Circuit, ControlSpec
    from vguitar.config import Config
    from vguitar.data import Dataset


# The input source line every circuit declares verbatim (see circuits/base.py).
# We rewrite it to an *external* source so ngspice queries get_vsrc_data for it.
_VIN_RE = re.compile(r"^\s*Vin\s+in\s+0\s+dc\s+0\s*$", re.IGNORECASE | re.MULTILINE)


def _import_error_message() -> str:
    return (
        "ngspice shared library is not available. Install it with:\n"
        "    uv run pyspice-post-installation --force-install-ngspice-dll"
    )


def _build_external_source_netlist(
    circuit: Circuit, params: dict[str, float] | None = None
) -> str:
    """Return the circuit netlist with ``Vin`` turned into an external source.

    The base netlist declares the input exactly as ``Vin in 0 dc 0``. ngspice's
    external-source syntax is the trailing ``external`` keyword; with it, ngspice
    invokes ``get_vsrc_data`` (node name ``"vin"``) at every solver timestep.
    A title line is required by SPICE (the first line is always the title).

    ``params`` selects netlist-mode control values (a tone cap, feedback resistor,
    bias voltage, ...) via :meth:`Circuit.netlist_for`; ``None`` renders defaults.
    """
    body = circuit.netlist_for(params)
    new_body, n = _VIN_RE.subn("Vin in 0 dc 0 external", body)
    if n == 0:
        raise ValueError(
            f"circuit {circuit.name!r} netlist must declare the input exactly as "
            "'Vin in 0 dc 0' (see vguitar.circuits.base)"
        )
    return f"* vguitar simulation: {circuit.name}\n{new_body}"


# Process-wide singletons. ngspice's shared library is fundamentally one
# instance per process: its cffi ``cdef`` can only run once, so constructing
# ``NgSpiceShared`` a second time raises "duplicate declaration of struct
# ngcomplex". We build ONE subclass and ONE instance and reuse them across all
# simulate() calls, swapping the input signal in via ``set_input`` each time.
# Typed as Any: PySpice/ngspice is an untyped C binding (no stubs), and our
# instance is a runtime-defined subclass with an extra set_input() method.
_NG_CLASS: Any = None
_NG: Any = None  # the singleton instance


def _get_shared_class() -> Any:
    """Define (once) and return the ``NgSpiceShared`` subclass for our callback."""
    global _NG_CLASS
    if _NG_CLASS is not None:
        return _NG_CLASS
    try:
        from PySpice.Spice.NgSpice.Shared import NgSpiceShared
    except ImportError as exc:  # ngspice DLL / PySpice missing
        raise RuntimeError(_import_error_message()) from exc

    class _ExternalSourceNg(NgSpiceShared):
        """Feeds a pre-upsampled input via get_vsrc_data; silences ngspice chatter."""

        _xs: np.ndarray = np.zeros(1, dtype=np.float64)  # replaced by set_input
        _sim_sr: int = 1
        _last_idx: int = 0

        def set_input(self, xs: np.ndarray, sim_sr: int) -> None:
            """Point the callback at the next signal (reused across simulations)."""
            self._xs = xs
            self._sim_sr = sim_sr
            self._last_idx = len(xs) - 1

        def get_vsrc_data(
            self, voltage: list[float], time: float, node: str, ngspice_id: int
        ) -> int:
            # Nearest-sample read on the uniform sim_sr grid; clamp to bounds so
            # the solver's look-ahead past the stop time stays well-defined.
            idx = round(time * self._sim_sr)
            if idx < 0:
                idx = 0
            elif idx > self._last_idx:
                idx = self._last_idx
            voltage[0] = float(self._xs[idx])
            return 0

        def send_char(self, message: str, ngspice_id: int) -> int:
            # Swallow ngspice's stdout/stderr log lines (otherwise very noisy).
            return 0

    _NG_CLASS = _ExternalSourceNg
    return _NG_CLASS


def _get_ng() -> Any:
    """Return the process-wide ngspice instance, creating it exactly once.

    ``new_instance`` caches by ``ngspice_id`` so the cffi ``cdef`` runs only on
    the first call — the key to running many circuits in one process.
    """
    global _NG
    if _NG is None:
        cls = _get_shared_class()
        _NG = cls.new_instance(ngspice_id=0, send_data=False)
    return _NG


def _options_lines(cfg: SimConfig) -> str:
    """SPICE ``.options`` for robust convergence of stiff nonlinear circuits.

    ``gmin`` stepping and ``method=gear`` (2nd-order implicit, A-stable) help
    nonlinear stages converge and damp the spurious ringing trapezoidal
    integration can produce at sharp clipping transitions.
    """
    return (
        f".options reltol={cfg.reltol:g} abstol={cfg.abstol:g} temp={cfg.temperature_c:g}\n"
        ".options gmin=1e-12 method=gear\n"
    )


def simulate(
    circuit: Circuit,
    x: np.ndarray,
    sr: int,
    cfg: SimConfig | None = None,
    *,
    params: dict[str, float] | None = None,
) -> np.ndarray:
    """Run ``x`` (at ``sr``) through ``circuit`` in ngspice; return its output.

    The input is upsampled to ``cfg.sim_sr`` and streamed sample-by-sample into
    an external voltage source via the ``get_vsrc_data`` callback (see module
    docstring). The transient result on ngspice's adaptive grid is resampled
    back to ``sr``, anti-alias-decimated so the target is band-limited.

    Args:
        circuit: the circuit to simulate (provides the netlist).
        x: input signal in volts, shape ``(N,)``.
        sr: sample rate of ``x`` (and of the returned output).
        cfg: simulation settings; defaults to :class:`SimConfig`.
        params: netlist-mode control values (e.g. a tone cap or feedback
            resistor) substituted via :meth:`Circuit.netlist_for`; ``None``
            renders the circuit's default netlist. Each combination is a fresh
            ``load_circuit`` -- which is already the per-call cost model, so a
            per-setting reload adds no measurable overhead over the ``.tran`` solve.

    Returns:
        The circuit output at node ``out``, float32, **exactly** ``len(x)``
        samples (padded/truncated to align with the input).

    Raises:
        RuntimeError: if the ngspice shared library cannot be loaded.
    """
    cfg = cfg or SimConfig()
    # Lazy import: resample is a sibling module; keep this module importable even
    # if it is built later, and avoid an import cycle.
    from vguitar.resample import resample, to_audio

    x = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    stop_s = n / sr
    # Pre-upsample once to the uniform sim grid for cheap, accurate callback reads.
    xs = np.ascontiguousarray(resample(x, sr, cfg.sim_sr), dtype=np.float64)

    # Reuse the process-wide instance; point its callback at this run's signal.
    ng = _get_ng()
    ng.set_input(xs, cfg.sim_sr)

    netlist = (
        _build_external_source_netlist(circuit, params)
        + "\n"
        + _options_lines(cfg)
        # No 'uic': let ngspice solve the DC operating point first so a biased
        # stage starts from its steady bias (coupling/bypass caps pre-charged),
        # instead of a long startup transient that would corrupt the dataset.
        + f".tran {cfg.max_step_s:g} {stop_s:g}\n"
        + ".end\n"
    )

    try:
        ng.load_circuit(netlist)
        ng.run()
        plot = ng.plot(None, ng.last_plot)
    except Exception as exc:  # PySpice/ngspice runtime errors -> clear message
        raise RuntimeError(f"ngspice simulation of {circuit.name!r} failed: {exc}") from exc

    if "time" not in plot or "out" not in plot:
        raise RuntimeError(
            f"ngspice produced no 'time'/'out' vector for {circuit.name!r}; "
            f"got {sorted(plot.keys())}"
        )

    # Vectors come back complex128 for AC-capable analyses; take the real part.
    t_out = np.ascontiguousarray(plot["time"]._data.real, dtype=np.float64)
    y_out = np.ascontiguousarray(plot["out"]._data.real, dtype=np.float64)

    # Drop this run's circuit + plot so the reused instance stays clean and does
    # not accumulate state across many circuits. Best-effort.
    with contextlib.suppress(Exception):
        ng.remove_circuit()
    with contextlib.suppress(Exception):
        ng.destroy("all")

    # Resample the irregular sim grid down to the audio rate (band-limited).
    y = to_audio(t_out, y_out, cfg.sim_sr, sr)
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    # Guarantee exact length alignment with the input.
    if y.shape[0] < n:
        y = np.pad(y, (0, n - y.shape[0]))
    elif y.shape[0] > n:
        y = y[:n]
    return np.ascontiguousarray(y, dtype=np.float32)


def make_dataset(circuit: Circuit, cfg: Config) -> Dataset:
    """Generate a training :class:`Dataset` by simulating a designed excitation.

    Builds the multi-level excitation (``vguitar.signals``), runs it through the
    circuit, and packages the aligned ``(x, y)`` pair with provenance metadata.

    Args:
        circuit: the circuit to characterize.
        cfg: a top-level :class:`vguitar.config.Config`.

    Returns:
        A :class:`vguitar.data.Dataset` at ``cfg.data.sr``.
    """
    # Lazy imports: keep this gating module importable before siblings exist.
    from vguitar.data import Dataset
    from vguitar.signals import build_training_excitation

    x = build_training_excitation(cfg.data)
    y = simulate(circuit, x, cfg.data.sr, cfg.sim)
    return Dataset(
        x,
        y,
        cfg.data.sr,
        name=circuit.name,
        meta={
            "circuit": circuit.name,
            "sim_sr": cfg.sim.sim_sr,
            "netlist_len": len(circuit.netlist()),
        },
    )


_DEFAULT_DI_PATH = "assets/guitar_di_loop.wav"


def _di_window(di: np.ndarray, n: int, offset_seed: int) -> np.ndarray:
    """A unit-peak ``n``-sample window of the DI loop at a deterministic offset."""
    di = np.asarray(di, dtype=np.float32).reshape(-1)
    if di.size == 0 or n <= 0:
        return di[:n].astype(np.float32)
    if di.size < n:
        di = np.tile(di, int(np.ceil(n / di.size)))
    off = (offset_seed * 7919) % (di.size - n + 1)
    w = di[off : off + n]
    p = float(np.max(np.abs(w))) or 1.0
    return (w / p).astype(np.float32)


def make_control_dataset(
    circuit: Circuit,
    grid: np.ndarray,
    control_specs: list[ControlSpec],
    cfg: Config | None = None,
    *,
    seg_dur_s: float = 3.0,
    seed: int = 0,
    excitation: np.ndarray | None = None,
    di_mix: float = 0.0,
    di_path: str | None = None,
    name: str | None = None,
    meta: dict[str, Any] | None = None,
) -> Dataset:
    """Generate a CONDITIONED dataset over arbitrary control axes.

    Each row of ``grid`` is one segment rendered at one fixed control setting.
    Columns are split by their :class:`ControlSpec` ``mode``:

    * ``pregain`` axes scale the input into the stage (the drive knob of a real
      overdrive pedal). Several pregain axes multiply together.
    * ``netlist`` axes substitute a value into the netlist via
      :meth:`Circuit.netlist_for` (a tone cap, a feedback resistor, a bias).

    For every row the model input is the *dry* excitation and the target is the
    circuit's response to ``g_pregain * input`` under the row's netlist params.
    The dense ``(N, C)`` control matrix records the raw control values so the
    model conditions on them directly; interpolation to unseen settings is tested
    by holding rows out of ``grid``.

    Args:
        circuit: circuit to characterize.
        grid: ``(S, C)`` control rows (a ``(S,)`` vector is treated as one column).
        control_specs: one :class:`ControlSpec` per column (length ``C``).
        cfg: pipeline config (defaults to :class:`Config`).
        seg_dur_s: duration of each per-setting excitation chunk.
        seed: base RNG seed (offset per segment for distinct synthetic excitations).
        excitation: optional shared unit-peak dry base used for *every* row
            (e.g. a guitar DI loop); when ``None``, each row gets its own rich
            synthetic excitation.
        di_mix: fraction of rows (0..1) rendered with a real guitar-DI window as
            the base instead of synthetic excitation. Putting real playing in the
            training data closes the synthetic-vs-real generalization gap. Ignored
            when ``excitation`` is given.
        di_path: path to the guitar-DI loop (defaults to the bundled asset).
        name, meta: dataset label / provenance (sensible defaults if omitted).

    Returns:
        A conditioned :class:`Dataset` with one control column per spec.
    """
    from dataclasses import replace
    from pathlib import Path

    from vguitar.config import Config
    from vguitar.data import Dataset
    from vguitar.signals import build_training_excitation, load_di

    grid = np.asarray(grid, dtype=np.float32)
    if grid.ndim == 1:
        grid = grid.reshape(-1, 1)
    if grid.ndim != 2 or grid.shape[0] == 0:
        raise ValueError(f"grid must be a non-empty (S, C) array, got shape {grid.shape}")
    n_cols = grid.shape[1]
    if len(control_specs) != n_cols:
        raise ValueError(
            f"control_specs has {len(control_specs)} entries but grid has {n_cols} columns"
        )

    cfg = cfg or Config()
    sr = cfg.data.sr
    pregain_idx = [i for i, s in enumerate(control_specs) if s.mode == "pregain"]
    netlist_idx = [i for i, s in enumerate(control_specs) if s.mode == "netlist"]

    # Choose which rows use a real-DI base (spread evenly across the grid).
    di_full: np.ndarray | None = None
    di_rows: set[int] = set()
    if excitation is None and di_mix > 0.0:
        path = di_path or _DEFAULT_DI_PATH
        if Path(path).exists():
            di_full = load_di(path, sr, peak=1.0)
            k = max(1, round(di_mix * len(grid)))
            di_rows = set(np.linspace(0, len(grid) - 1, k).round().astype(int).tolist())

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    bounds: list[int] = [0]
    values: list[list[float]] = []
    for j, row in enumerate(grid):
        if excitation is not None:
            base = np.asarray(excitation, dtype=np.float32).reshape(-1)
        elif j in di_rows and di_full is not None and di_full.size:
            base = _di_window(di_full, round(seg_dur_s * sr), offset_seed=seed + j)
        else:
            # Unit-peak, content-rich dry excitation; the pregain axes do the scaling.
            dcfg = replace(cfg.data, duration_s=seg_dur_s, drive_levels=(1.0,), seed=seed + j)
            base = build_training_excitation(dcfg)
        g = float(np.prod([row[i] for i in pregain_idx])) if pregain_idx else 1.0
        params = {control_specs[i].name: float(row[i]) for i in netlist_idx} or None
        y = simulate(circuit, (g * base).astype(np.float32), sr, cfg.sim, params=params)
        xs.append(base)
        ys.append(y)
        values.append([float(v) for v in row])
        bounds.append(bounds[-1] + len(base))

    return Dataset.from_segments(
        np.concatenate(xs),
        np.concatenate(ys),
        sr,
        bounds,
        np.asarray(values, dtype=np.float32),
        name=name or f"{circuit.name}_ctl",
        meta=meta
        or {
            "circuit": circuit.name,
            "controls": [s.name for s in control_specs],
            "modes": [s.mode for s in control_specs],
        },
        control_names=[s.name for s in control_specs],
        control_kinds=[s.kind for s in control_specs],
    )


def make_drive_dataset(
    circuit: Circuit,
    drive_values: list[float],
    cfg: Config | None = None,
    *,
    seg_dur_s: float = 3.0,
    seed: int = 0,
    di_mix: float = 0.0,
    di_path: str | None = None,
) -> Dataset:
    """Generate a CONDITIONED dataset for one continuous "drive" control.

    A thin wrapper over :func:`make_control_dataset` for the common single-axis
    pre-gain case (a real overdrive pedal's drive knob): the model input is the
    *dry* excitation; the target is the circuit driven at ``g * input`` for each
    ``g`` in ``drive_values``. Interpolation to unseen ``g`` is tested by holding
    values out of ``drive_values``.

    Args:
        circuit: circuit to characterize.
        drive_values: the drive settings ``g`` to simulate (one segment each).
        cfg: pipeline config (defaults to :class:`Config`).
        seg_dur_s: duration of each per-setting excitation chunk.
        seed: base RNG seed (offset per segment for distinct excitations).

    Returns:
        A conditioned :class:`Dataset` with a single ``"drive"`` control column.
    """
    from vguitar.circuits.base import ControlSpec

    drives = [float(g) for g in drive_values]
    grid = np.asarray([[g] for g in drives], dtype=np.float32)
    spec = ControlSpec(
        name="drive",
        kind="continuous",
        lo=min(drives),
        hi=max(drives),
        default=drives[0],
        mode="pregain",
    )
    return make_control_dataset(
        circuit,
        grid,
        [spec],
        cfg,
        seg_dur_s=seg_dur_s,
        seed=seed,
        di_mix=di_mix,
        di_path=di_path,
        name=f"{circuit.name}_drive",
        meta={"circuit": circuit.name, "control": "drive", "drive_values": drives},
    )
