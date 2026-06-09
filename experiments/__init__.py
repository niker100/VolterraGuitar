"""Research experiments + their visualizations (kept out of the shipped package).

Each script here is a reproducible probe/A-B used to make a design decision; the
shared boilerplate (timestamped file logging, held-ESR eval, dataset loading) lives
in :mod:`experiments.common`, and every result figure is rendered in the project
house style by :mod:`experiments.figures` (regenerate all with
``uv run python -m experiments.figures``). Results land in ``outputs/`` (gitignored,
regenerable); the scripts are the source of truth.
"""
