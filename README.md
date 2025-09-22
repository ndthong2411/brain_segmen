# Brain Tumor Segmentation

## Overview
BrainTumNet is a PyTorch-based workflow for multi-task brain tumor analysis on BraTS2020 MRI slices. The model couples semantic segmentation with tumor-grade classification, using configurable data transforms, cosine-annealed training, and mixed precision support. Scripts and configs in this repository reproduce the end-to-end preprocessing, training, and evaluation pipeline.

## Project Structure
- `braintumnet/src/braintumnet/` core package (`data`, `engine`, `models`, `utils`, plus `losses.py` and `metrics.py`).
- `braintumnet/scripts/` command-line tools for data prep, training, evaluation, and batch visualization.
- `braintumnet/configs/` YAML experiment files; start from `default.yaml` and override as needed.
- `braintumnet/data/` expected home for raw and processed BraTS assets (git-ignored by default).
- `braintumnet/tests/` placeholder for unit and smoke tests.
- `main.ipynb` exploratory notebook for quick experimentation or visualization.

## Getting Started
1. Create an isolated environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
   pip install -r braintumnet/requirements.txt
   ```
2. Prepare data (download BraTS2020 manually, then):
   ```bash
   python braintumnet/scripts/prepare_brats2020.py --raw /path/to/BraTS2020/TrainingData --out braintumnet/data/processed
   ```
   This generates normalized slices and cross-validation splits under `data/processed`.
3. Train a fold:
   ```bash
   python braintumnet/scripts/train.py --cfg braintumnet/configs/default.yaml --fold 0
   ```
   Checkpoints land in `checkpoints/`; TensorBoard logs in `runs/` when enabled.
4. Evaluate or visualize results:
   ```bash
   python braintumnet/scripts/evaluate.py --ckpt checkpoints/braintumnet_best_fold0.pth --cfg braintumnet/configs/default.yaml
   python braintumnet/scripts/visualize_batch.py --cfg braintumnet/configs/default.yaml --fold 0
   ```

## Configuration
Adjust paths, augmentation, and model hyperparameters in `configs/default.yaml`. Override values via CLI flags or by copying the file into a new experiment config committed alongside your runs. Keep dataset locations outside the repo root when working with sensitive data.

## Contributing
See `AGENTS.md` for contributor conventions covering structure, coding standards, testing expectations, and pull request etiquette.

## License
Released under the MIT License. See `LICENSE` for details.
