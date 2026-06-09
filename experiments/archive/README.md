# Archived experiment phases

Completed one-shot studies whose verdicts are folded into the docs and the model.
Kept runnable for reproduction (`uv run python -m experiments.archive.<name>`), but
nothing active imports from here. Results live in `outputs/`, decisions in `docs/`.

| phase | scripts | verdict (where recorded) |
|---|---|---|
| Mixed-activation block | `radical_mixedact_prod`, `make_mixedact_fair_ab`, `mixed_vs_gated_final`, `mixed_default_retrain` | mixed tanh/gelu/relu/abs/snake block adopted into CIRCE3 (`docs/optimal-architecture.md`) |
| Architecture ablations | `radical_arch_ablation`, `make_arch_ablation_plot`, `radical_frontier_test`, `sweep_uniform`, `make_radical_leaderboard`, `make_lever_plot` | CIRCE3 input-scaling TCN frontier (`docs/CIRCE3-modelcard.md`) |
| Oversampling | `radical_os_ablation`, `make_oversample_figure` | OS2 adopted; OS>2 measured worse (`docs/sota-campaign.md` dead-ends) |
| Grad-clip | `radical_gradclip_test`, `radical_gradclip_rest`, `make_gradclip_plot` | clip=1.0 confirmed optimal |
| Generalization A/B | `radical_generalize_ab` | raw-scored TCN cross-check used in the DC-blocker diagnosis |
| Wavefolder attempts | `wavefolder_os_ab`, `wavefolder_shaper_ab`, `radical_wh_greybox3` | OS/shaper inert; grey-box PWL bank (`PWLBank`) is the reusable piece (`docs/wavefolder-frontier-plan.md`) |
| Circuit memory audit | `analyze_circuit_memory` | hysteretic_fuzz τ = 22–440 ms ≫ TCN receptive field → IIR state lever |
| Spectral branch | `spectral_probe`, `spectral_slim`, `spectral_only_probe`, `spectral_leads`, `spectral_zoo_run`, `spectral_zoo/` | spectral hybrids gave no uniform win; dropped from the SOTA line (`outputs/figs/frontier/spectral_*.png`) |
