"""Cross-circuit validation: CIRCE (and optional baselines) over a set of circuits.

Where ``vguitar circe`` validates ONE circuit in depth, this aggregates the
quantitative core (:func:`vguitar.benchmark.circe_eval._validate_one`) across many
circuits to answer the headline question: does the approach hold up as the
circuits get more complex? It emits:

* a per-circuit CIRCE summary (trained vs held-out ESR, moving-knob RTF,
  streaming error, params, #controls), and
* two figures: a cross-circuit summary (ESR + RTF per circuit) and a
  circuit x model ESR matrix (CIRCE plus any requested unconditioned baselines,
  each trained on that circuit's native dataset).

Reproducible via ``vguitar validate --circuits ... [--models ...]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from vguitar.config import Config


def _baseline_esr(circuit_name: str, model_name: str, cfg: Config) -> float:
    """Test ESR of one unconditioned baseline trained on the circuit's dataset."""
    from vguitar.benchmark.run import _evaluate, _instantiate, _load_or_make_dataset

    ds = _load_or_make_dataset(circuit_name, cfg)
    train, val, test = ds.split(cfg.train.val_fraction, cfg.train.test_fraction)
    model = _instantiate(model_name, cfg.train.device)
    res = _evaluate(model_name, model, train, val, test, cfg)
    return float(res["esr"])


def run_validation(
    circuit_names: list[str],
    *,
    models: tuple[str, ...] = ("circe",),
    cfg: Config | None = None,
    retrain: bool = False,
    regen: bool = False,
    seg_dur_s: float = 2.0,
    epochs: int = 150,
    channels: int = 24,
    di_mix: float | None = None,
) -> dict[str, Any]:
    """Validate CIRCE across ``circuit_names``; aggregate + plot the comparison.

    Args:
        circuit_names: circuits to validate (each via the CIRCE control sweep).
        models: columns of the circuit x model ESR matrix. Always includes CIRCE
            (its trained-mean ESR); any other names are unconditioned baselines
            trained per circuit on the native dataset.
        cfg: pipeline config.
        retrain, regen: force model retraining / dataset regeneration.
        seg_dur_s, epochs, channels: passed through to the per-circuit CIRCE fit.

    Returns:
        ``{"summary": [per-circuit dicts], "matrix": ndarray, "circuits": [...],
        "models": [...], "figures": [paths]}``.
    """
    import matplotlib.pyplot as plt
    from rich.console import Console
    from rich.table import Table

    from vguitar import plotting as plot
    from vguitar.benchmark.circe_eval import _validate_one
    from vguitar.config import Config

    cfg = cfg or Config()
    cfg.paths.ensure()
    console = Console()

    # CIRCE per circuit (the quantitative core).
    summary: list[dict[str, Any]] = []
    circe_held: dict[str, float] = {}
    circe_trained: dict[str, float] = {}
    for cname in circuit_names:
        console.print(f"[bold]validating[/] {cname} ...")
        res = _validate_one(
            cname, cfg=cfg, retrain=retrain, regen=regen, seg_dur_s=seg_dur_s,
            epochs=epochs, channels=channels, di_mix=di_mix, probe_thd=False, console=console,
        )
        interp = res["interp"]
        summary.append({
            "circuit": cname, "n_controls": len(res["specs"]),
            "trained_esr": interp["trained_mean"], "held_mean": interp["held_mean"],
            "held_worst": interp["held_worst"], "rtf_c": res["rtf_c"]["rtf"],
            "rtf_m": res["rtf_m"]["rtf"], "streaming": res["streaming_err"],
            "moving": res["moving_err"], "params": res["params"],
        })
        circe_trained[cname] = interp["trained_mean"]
        circe_held[cname] = interp["held_mean"]

    # Circuit x model ESR matrix (CIRCE = trained-mean; baselines trained natively).
    model_cols = ["circe", *[m for m in models if m != "circe"]]
    matrix = np.full((len(circuit_names), len(model_cols)), np.nan, dtype=np.float64)
    for i, cname in enumerate(circuit_names):
        for j, mname in enumerate(model_cols):
            if mname == "circe":
                matrix[i, j] = circe_trained[cname]
            else:
                try:
                    matrix[i, j] = _baseline_esr(cname, mname, cfg)
                except Exception as exc:  # a baseline must not abort the sweep
                    console.print(f"  [yellow]{mname} on {cname} failed: {exc}[/]")

    # --- report ---
    table = Table(title="CIRCE cross-circuit validation")
    for col in ("circuit", "#ctrl", "trained ESR", "held ESR", "held worst",
                "RTF const", "RTF moving", "stream", "params"):
        table.add_column(col, justify="right" if col != "circuit" else "left")
    for r in summary:
        table.add_row(
            r["circuit"], str(r["n_controls"]), f"{r['trained_esr']:.4f}",
            f"{r['held_mean']:.4f}", f"{r['held_worst']:.4f}",
            f"{r['rtf_c']:.1f}x", f"{r['rtf_m']:.1f}x", f"{r['streaming']:.1e}",
            f"{r['params']:,}",
        )
    console.print(table)

    # --- figures ---
    outdir = cfg.paths.outputs / "figs"
    outdir.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []

    def _save(fig: Any, stem: str) -> None:
        p = outdir / f"validate_{stem}.png"
        fig.savefig(p)
        plt.close(fig)
        figures.append(str(p))

    _save(plot.fig_cross_circuit_summary(summary, name=""), "cross_circuit")
    _save(plot.fig_circuit_model_esr(matrix, list(circuit_names), model_cols, name=""), "circuit_model_esr")
    console.print(f"wrote {len(figures)} figures to {outdir}")

    return {"summary": summary, "matrix": matrix, "circuits": list(circuit_names),
            "models": model_cols, "figures": figures}
