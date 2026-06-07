"""Control-grid sampling for conditioned (multi-control) datasets.

Generating a conditioned dataset means rendering the circuit at a set of control
settings. For one or two knobs a **full-factorial** grid is best -- it is regular,
so the validation heatmaps have clean axes. For three or more knobs the factorial
product explodes (``n_axis ** C``), so we switch to a low-discrepancy **Sobol**
sequence (``scipy.stats.qmc``) that covers the control hypercube far more evenly
than random sampling for a fixed simulation budget.

Each control axis is a :class:`vguitar.circuits.base.ControlSpec`; the sampler
scales the unit hypercube to ``[lo, hi]`` per axis and rounds ``discrete``
(switch) axes to integers. ``holdout_grid`` draws *unseen* settings -- points kept
only if they are far enough (in normalized control space) from every trained
point -- which is how interpolation is tested honestly.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vguitar.circuits.base import ControlSpec


def _round_discrete(grid: np.ndarray, specs: list[ControlSpec]) -> np.ndarray:
    """Round ``discrete`` (switch) axes to integers, clamped to ``[lo, hi]``."""
    out = np.array(grid, dtype=np.float32, copy=True)
    for i, s in enumerate(specs):
        if s.kind == "discrete":
            out[:, i] = np.clip(np.round(out[:, i]), s.lo, s.hi)
    return out


def _factorial(specs: list[ControlSpec], n_axis: int) -> np.ndarray:
    """Full-factorial product of ``n_axis`` points per axis -> ``(n_axis**C, C)``."""
    axes = [np.linspace(s.lo, s.hi, n_axis, dtype=np.float64) for s in specs]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid = np.stack([m.reshape(-1) for m in mesh], axis=1).astype(np.float32)
    return _round_discrete(grid, specs)


def _sobol(specs: list[ControlSpec], n: int, seed: int) -> np.ndarray:
    """``n`` Sobol points scaled to each axis range -> ``(n, C)``."""
    from scipy.stats import qmc

    with warnings.catch_warnings():
        # Sobol warns when n is not a power of two (balance property); we accept
        # arbitrary n on purpose, so silence the cosmetic warning.
        warnings.simplefilter("ignore")
        unit = qmc.Sobol(d=len(specs), scramble=True, seed=seed).random(n)
    lo = np.array([s.lo for s in specs], dtype=np.float64)
    hi = np.array([s.hi for s in specs], dtype=np.float64)
    grid = (lo + unit * (hi - lo)).astype(np.float32)
    return _round_discrete(grid, specs)


def control_grid(
    specs: list[ControlSpec],
    *,
    n_axis: int = 5,
    budget: int | None = None,
    mode: str = "auto",
    seed: int = 0,
) -> np.ndarray:
    """Return an ``(S, C)`` grid of control settings to simulate.

    Args:
        specs: one :class:`ControlSpec` per control axis.
        n_axis: points per axis for the factorial mode.
        budget: hard cap on the number of rows ``S`` (the simulation budget).
            For factorial mode a grid exceeding ``budget`` is an error; for Sobol
            mode it is the number of points drawn (default 32).
        mode: ``"factorial"``, ``"sobol"``, or ``"auto"`` (factorial when
            ``n_axis ** C`` fits the budget, else Sobol).
        seed: Sobol scramble seed.

    Returns:
        ``(S, C)`` float32 grid; ``discrete`` axes are integer-valued.
    """
    if not specs:
        raise ValueError("control_grid needs at least one ControlSpec")
    c = len(specs)
    if mode == "auto":
        mode = "factorial" if n_axis**c <= (budget or 36) else "sobol"

    if mode == "factorial":
        grid = _factorial(specs, n_axis)
        if budget is not None and grid.shape[0] > budget:
            raise ValueError(
                f"factorial grid has {grid.shape[0]} rows > budget {budget}; "
                f"reduce n_axis or use mode='sobol'"
            )
        return grid
    if mode == "sobol":
        return _sobol(specs, budget or 32, seed)
    raise ValueError(f"unknown mode {mode!r}; use 'factorial', 'sobol', or 'auto'")


def _normalized(grid: np.ndarray, specs: list[ControlSpec]) -> np.ndarray:
    """Map each axis to ``[0, 1]`` by its declared span (for distance metrics)."""
    lo = np.array([s.lo for s in specs], dtype=np.float64)
    hi = np.array([s.hi for s in specs], dtype=np.float64)
    span = np.where(hi > lo, hi - lo, 1.0)
    return (np.asarray(grid, dtype=np.float64) - lo) / span


def nn_distance(points: np.ndarray, trained: np.ndarray, specs: list[ControlSpec]) -> np.ndarray:
    """Per-point nearest-neighbour L2 distance to ``trained`` in normalized space.

    Returns one distance per row of ``points``; ``inf`` if ``trained`` is empty.
    This is the multi-control generalization of "distance to nearest trained
    setting" used by the interpolation plots.
    """
    pts = _normalized(points, specs)
    if trained is None or len(trained) == 0:
        return np.full(pts.shape[0], np.inf)
    tr = _normalized(trained, specs)
    # (P, T) pairwise distances -> min over T.
    d = np.linalg.norm(pts[:, None, :] - tr[None, :, :], axis=2)
    return d.min(axis=1)


def holdout_grid(
    specs: list[ControlSpec],
    trained: np.ndarray,
    *,
    n: int = 3,
    seed: int = 1,
    min_dist: float = 0.1,
    oversample: int = 16,
) -> np.ndarray:
    """Sample ``n`` *held-out* settings far from every trained point.

    Draws ``n * oversample`` Sobol candidates, keeps those whose normalized
    nearest-neighbour distance to ``trained`` is at least ``min_dist``, and
    returns the ``n`` most-distant of those (so held-out points genuinely probe
    interpolation, never coincide with a trained setting).

    Args:
        specs: one :class:`ControlSpec` per axis.
        trained: ``(T, C)`` trained control rows to stay away from.
        n: number of held-out rows to return.
        seed: Sobol seed (use a different seed than the trained grid).
        min_dist: minimum normalized NN-distance for a candidate to qualify.
        oversample: candidate pool multiplier.

    Returns:
        ``(<=n, C)`` float32 grid of held-out settings (fewer than ``n`` only if
        too few candidates clear ``min_dist``).
    """
    cand = _sobol(specs, max(n * oversample, n), seed)
    dist = nn_distance(cand, trained, specs)
    qualified = np.where(dist >= min_dist)[0]
    # Prefer points clearing min_dist; if none qualify, fall back to the whole
    # pool. Either way, return the most-distant n (descending NN-distance).
    pool = qualified if qualified.size else np.arange(dist.size)
    order = pool[np.argsort(dist[pool])[::-1]]
    return cand[order[:n]]
