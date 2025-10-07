# Multimodal BraTS2020 Dataset Preprocessing

## Overview

Successfully processed the complete BraTS2020 dataset with **all 4 MRI modalities** for enhanced tumor segmentation.

## What Are the 4 Modalities?

BraTS2020 includes 4 different MRI sequences, each highlighting different tumor characteristics:

| Modality | Full Name | What It Shows | Best For |
|----------|-----------|---------------|----------|
| **FLAIR** | Fluid Attenuated Inversion Recovery | Edema (fluid buildup) | Detecting tumor-induced edema |
| **T1** | T1-weighted | Anatomical structure | Brain anatomy reference |
| **T1CE** | T1 Contrast-Enhanced | Active tumor core | Tumor enhancement/core |
| **T2** | T2-weighted | Whole tumor region | Complete tumor extent |

### Why Use All 4 Together?

- **Complementary information**: Each modality highlights different aspects
- **Better accuracy**: Research shows **5-10% IoU improvement** over single-modal
- **More robust**: Model learns from multiple perspectives
- **Clinical standard**: Radiologists use all 4 modalities for diagnosis

## Processing Results

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total H5 files processed** | 57,195 |
| **Slices with tumor** | 22,677 |
| **Unique patients** | 369 |
| **Image format** | `.npy` (256×256×4) |
| **Data type** | float32 |
| **File size per image** | ~1MB (vs ~70KB for single-modal PNG) |

### Data Format

**Single-modal (T1CE only)**:
- File: `vol1_slice100.png`
- Shape: `(256, 256)` - grayscale image
- Channels: 1 (T1CE only)
- Size: ~70KB

**Multimodal (all 4)**:
- File: `vol1_slice100.npy`
- Shape: `(256, 256, 4)` - 4-channel tensor
- Channels: [FLAIR, T1, T1CE, T2] in order
- Size: ~1MB

### Cross-Validation Splits

```
Fold 0: 295 train (18,102 slices) | 74 val (4,573 slices)
Fold 1: 295 train (17,967 slices) | 74 val (4,710 slices)
Fold 2: 295 train (18,272 slices) | 74 val (4,405 slices)
Fold 3: 295 train (18,078 slices) | 74 val (4,599 slices)
Fold 4: 296 train (18,288 slices) | 73 val (4,389 slices)
```

## Dataset Locations

### Single-Modal (T1CE only)
```
braintumnet/data/processed_full/
├── images/        22,677 PNG files (256×256×1)
├── masks/         22,677 PNG files
├── labels.csv     369 patients
└── mapping.csv    22,677 slice mappings
```

### Multimodal (All 4 sequences)
```
braintumnet/data/processed_full_multimodal/
├── images/        22,677 NPY files (256×256×4)  ← 4 channels!
├── masks/         22,677 PNG files
├── labels.csv     369 patients
└── mapping.csv    22,677 slice mappings
```

## Comparison: Single vs Multi-Modal

| Aspect | Single-Modal (T1CE) | Multi-Modal (All 4) |
|--------|-------------------|-------------------|
| **Input channels** | 1 | 4 |
| **File format** | PNG (image) | NPY (array) |
| **Disk space** | ~1.6 GB | ~23 GB |
| **GPU memory** | Lower (batch_size=16) | Higher (batch_size=12) |
| **Training speed** | Faster | ~20% slower |
| **Expected IoU** | 0.60-0.70 | **0.65-0.75** ✓ |
| **Model complexity** | Simpler | More parameters |

## Training Recommendations

### Option 1: Start with Single-Modal (FASTER)
**Best for**: Quick baseline, debugging, faster iteration

```bash
cd braintumnet
python scripts/train.py --config configs/full_dataset.yaml
```

- Training time: ~20-25 hours
- Expected IoU: 0.60-0.70
- Good for fixing the plateau issue first

### Option 2: Train with Multi-Modal (BETTER ACCURACY)
**Best for**: Maximum performance, final model

```bash
cd braintumnet
python scripts/train.py --config configs/full_dataset_multimodal.yaml
```

- Training time: ~24-30 hours
- Expected IoU: **0.65-0.75** (5-10% better)
- Requires more GPU memory (batch_size=12)

## Configuration Files

### Single-Modal Config
[configs/full_dataset.yaml](../configs/full_dataset.yaml)
```yaml
model:
  in_channels: 1          # T1CE only
train:
  batch_size: 16          # Fits comfortably in RTX 3090
```

### Multi-Modal Config
[configs/full_dataset_multimodal.yaml](../configs/full_dataset_multimodal.yaml)
```yaml
model:
  in_channels: 4          # All 4 modalities
train:
  batch_size: 12          # Reduced for 4-channel input
  amp: true               # CRITICAL for GPU memory
```

## Expected Performance Improvement

### Previous (13 patients, T1CE only)
- IoU: **0.44-0.46** (plateaued) ❌
- Dice: 0.62-0.63
- Issue: Severe overfitting

### Single-Modal Full Dataset (369 patients, T1CE)
- IoU: **0.60-0.70** ✓
- Dice: 0.75-0.82
- Improvement: +15-25 IoU points

### Multi-Modal Full Dataset (369 patients, all 4)
- IoU: **0.65-0.75** ✓✓
- Dice: 0.79-0.85
- Improvement: +20-30 IoU points
- **+5-10% better than single-modal**

## Technical Details

### Preprocessing Command Used
```bash
python scripts/prepare_brats2020_h5.py \
  --h5_root "E:\thong\code\brain_segmen\brats2020_data\bcs2020\archive\BraTS2020_training_data\content\data" \
  --meta_csv "E:\thong\code\brain_segmen\brats2020_data\bcs2020\archive\BraTS20 Training Metadata.csv" \
  --out data/processed_full_multimodal \
  --multimodal
```

### Key Parameters
- `--multimodal`: Save all 4 modalities as 4-channel arrays
- No `--max_slices`: Process ALL data
- `min_tumor_ratio=0.001`: Filter blank slices (default)

### Processing Time
- ~15-20 minutes on Windows system
- Processing speed: ~60-80 slices/second
- Slightly slower than single-modal due to 4x data

## Next Steps

### Recommended Workflow

1. **Start with single-modal** (T1CE only):
   ```bash
   python scripts/train.py --config configs/full_dataset.yaml
   ```
   - Faster training
   - Good baseline performance
   - Validates that full dataset fixes plateau

2. **Then try multi-modal** for better results:
   ```bash
   python scripts/train.py --config configs/full_dataset_multimodal.yaml
   ```
   - 5-10% performance boost
   - Final production model

### Monitor Training
```bash
tensorboard --logdir runs/
```

---

**Status**: Both datasets ready for training
**Created**: 2025-10-06
**Total processing time**: ~30-35 minutes (both datasets)
