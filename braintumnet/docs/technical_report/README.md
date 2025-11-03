# BrainTumNet Technical Report Series

Welcome! This “technical_report” folder distills everything you need to grasp the BrainTumNet project end-to-end. The goal is clarity: each document is short, practical, and walks from raw data to deployed models without assuming you have the rest of the repository open.

## How to use these reports

1. **Start here** to understand the overall scope and the document map.
2. Follow the files in numerical order; they mirror the lifecycle of a new experiment.
3. Each page ends with “Next Steps” so you can jump directly to the right topic.

| File | Focus | When to read |
| ---- | ----- | ------------ |
| `01_data_pipeline.md` | Raw → processed slices → LMDB. | Preparing data or verifying splits. |
| `02_models.md` | Architectures (SegUNetV2, BrainTumNetV2, Swin-UNETR, nnU-Net, TransUNet, LG-UNETR). | Evaluating which backbone to train. |
| `03_training_and_loss.md` | Losses, schedulers, amp, multi-task setup. | Before launching training jobs or editing configs. |
| `04_evaluation_inference.md` | Metrics, checkpoints, inference scripts, notebook tips. | Validating results and exporting predictions. |
| `05_operations_troubleshooting.md` | Practical issues (hardware, configs, NaNs, bad metrics). | When something breaks or performance looks odd. |
| `06_appendix_references.md` | Reference tables, CLI commands, key repo paths. | Quick lookup after you know the basics. |

## Guiding principles

- **Single source of truth:** Each report references actual code or configuration locations so you can cross-check facts quickly.
- **Accurate but readable:** Sentences are short, jargon is explained, and Vietnamese/English term pairs are included where helpful.
- **Actionable:** Every section mentions exact scripts (`python braintumnet/scripts/...`) and configuration blocks to edit.

You can open these files directly in any Markdown viewer. They follow the same style and are easy to diff if you extend them in the future.
