# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `braintumnet/src/braintumnet`, grouped into `data` for preprocessing utilities, `engine` for training loops, `models` for network definitions, and `utils` for shared helpers alongside `losses.py` and `metrics.py`. Command-line entry points sit in `braintumnet/scripts` (`train.py`, `evaluate.py`, `prepare_brats2020.py`, `visualize_batch.py`). Versioned configuration defaults are tracked in `braintumnet/configs/default.yaml`. Keep processed BraTS2020 slices under `braintumnet/data/processed` and raw archives in `brats2020_data/`; both locations are intentionally git-ignored.

## Build, Test, and Development Commands
Create an isolated environment, install dependencies, and ensure CUDA visibility where available:
- `python -m venv .venv && ./.venv/Scripts/activate`: set up a local virtualenv on Windows (adjust path for POSIX).
- `pip install -r braintumnet/requirements.txt`: sync runtime packages.
- `python braintumnet/scripts/prepare_brats2020.py --raw <BraTS2020 root> --out braintumnet/data/processed`: preprocess data and generate fold splits.
- `python braintumnet/scripts/train.py --cfg braintumnet/configs/default.yaml --fold 0`: train one fold and emit checkpoints to the configured logging directory.
- `python braintumnet/scripts/evaluate.py --ckpt <path> --cfg braintumnet/configs/default.yaml`: run metrics on held-out folds.

## Coding Style & Naming Conventions
Write Python with 4-space indentation, PEP 8 spacing, and `snake_case` names for modules, functions, and variables. Keep imports grouped (stdlib, third-party, local) and avoid wildcard imports. Use explicit type hints for public functions, mirror the existing f-string logging style, and factor substantial logic into the appropriate subpackage (e.g., dataloaders in `data`, optimizers in `engine`).

## Testing Guidelines
Add focused unit or smoke tests under `braintumnet/tests`, mirroring the package layout. Target `pytest` for new suites; invoke with `python -m pytest braintumnet/tests`. At minimum, cover critical preprocessing steps and metric computations; new training features should include deterministic smoke tests that exercise CPU paths. When touching data transforms, capture representative sample inputs in fixtures and assert expected tensor shapes and dtype.

## Commit & Pull Request Guidelines
Follow the existing concise, imperative commit style (e.g., "Add dice metric tests"). Reference related issues in the body when applicable, and ensure each commit remains scoped to a single concern. Pull requests should summarize motivation, list key changes, describe data or checkpoints required, and note how they were validated (commands run, datasets used). Attach screenshots or TensorBoard snippets when altering visualizations or metrics dashboards, and flag any compatibility-breaking configuration changes.

## Data & Configuration Tips
Configuration knobs belong in YAML files; avoid hard-coded paths in scripts. Keep secrets and patient data out of the repository; use environment variables or `.env` files ignored by git. Before committing, run scripts with a small sample slice set to confirm paths resolve on clean machines.
