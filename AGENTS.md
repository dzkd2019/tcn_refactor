# Repository Guidelines

## Project Background & Goals

This repository develops a stream-only Temporal Convolutional Network (TCN) for
PdCu hydrogen sensors. It estimates concentration from one sensor sample at a
time while compensating for temperature, humidity, noise, and baseline drift.
The system must remain causal: it cannot depend on future samples or batch
input. PdCu dynamics are not assumed strictly first order, so synthetic data
and models must support multiple response shapes.

## Project Structure & Module Organization

The executable entry point is `main.py`; command parsing lives in
`tcn_refactor/cli.py`. Core code is organized by responsibility under
`tcn_refactor/`: `model.py` defines the causal TCN, `features.py` owns the shared
feature pipeline, and `session.py` implements streaming state. `synthetic.py`,
`dataset.py`, `training.py`, and `metrics.py` cover experiments. Tests are in
`tests/`, design notes in `docs/`. Generated artifacts go in ignored `data/`,
`checkpoints/`, and `figures/` directories.

## Build, Test, and Development Commands

Use the locked `uv` environment:

- `uv sync` installs the Python 3.14 project dependencies.
- `uv run python -m unittest discover -s tests -v` runs all regression tests.
- `uv run ruff check .` runs import and compatibility checks.
- `uv run python -m compileall -q tcn_refactor main.py` performs a quick syntax check.
- `uv run python main.py generate --output data/scenarios.npz` creates synthetic data.
- `uv run python main.py train ... --device cuda` trains on CUDA; use
  `window-eval` for aligned windows and `replay` for the full deployment path.

## Coding Style & Naming Conventions

Use four-space indentation and type hints. Follow Python
`snake_case` for functions/variables, `PascalCase` for classes, and uppercase
constants. Keep docstrings and explanatory comments in Chinese; document tensor
shapes at boundaries (for example, `(B, 8, L) -> (B, L)`). Preserve causality:
features and predictions must never inspect future samples. Run Ruff before
submitting.

## Testing Guidelines

Tests use Python's `unittest`; name files `test_*.py` and methods `test_*`.
Add regression tests for each behavior change. Protect batch/stream feature
equivalence, output causality, `weight_norm`, safe NPZ loading, and session
state transitions. Accuracy changes should report independent `window-eval` and
`replay` metrics.

## Commit & Pull Request Guidelines

The history uses concise descriptive subjects such as `Initial commit: gas
sensor streaming TCN system`. Use short, imperative summaries, optionally scoped
(for example, `training: add relative Huber loss`). Keep generated artifacts out
of commits. Pull requests should state motivation, changed modules, validation
commands, relevant MAPE/detection results, and checkpoint or feature-version
incompatibilities. Link issues; include plots only when useful.
