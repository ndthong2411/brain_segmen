# BrainTumNet Phase 2 - Full Pipeline Notebook

## Tổng quan

File `fullmain.ipynb` là notebook executable hoàn chỉnh chứa toàn bộ code để chạy BrainTumNet Phase 2 từ đầu đến cuối, bao gồm:
- Data preprocessing
- Model architecture (SegUNetV2 Phase 2)
- Training pipeline
- Evaluation & metrics

## Model: SegUNetV2 Phase 2

**Tương đương với lệnh:**
```bash
python braintumnet/scripts/train.py --model segunetv2_phase2 --fold 4
```

**Key Features:**
- Multi-Scale Transformer Bottleneck (patch sizes: 4, 8, 16)
- Attention Gates in decoder
- Boundary Refinement Module (Phase 1)
- Deep Supervision
- Advanced medical augmentations

**Architecture:**
- Base channels: 64
- Transformer dim: 512
- Depth: 4 layers
- Attention heads: 8
- Total parameters: ~32-38M

## Cấu trúc thư mục yêu cầu

Trước khi chạy notebook, chuẩn bị cấu trúc thư mục như sau:

```
project_root/
├── data/
│   └── raw/
│       └── BraTS2020_TrainingData/
│           └── MICCAI_BraTS2020_TrainingData/
│               ├── BraTS20_Training_001/
│               ├── BraTS20_Training_002/
│               └── ...
├── fullmain.ipynb  (file notebook này)
└── (các thư mục sau sẽ được tạo tự động)
    ├── data/
    │   ├── processed_multiclass_4class/
    │   └── lmdb_processed_multiclass_4class/
    ├── logs/
    ├── runs/
    └── checkpoints/
```

## Hướng dẫn sử dụng

### 1. Chuẩn bị môi trường

```bash
# Cài đặt dependencies
pip install torch torchvision
pip install numpy pandas nibabel pillow scikit-learn scipy tqdm
pip install tensorboard
pip install lmdb  # optional
```

### 2. Chuẩn bị dữ liệu

- Download BraTS 2020 dataset
- Giải nén vào thư mục `data/raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/`
- Đảm bảo mỗi case có cấu trúc:
  ```
  BraTS20_Training_XXX/
  ├── BraTS20_Training_XXX_flair.nii.gz
  ├── BraTS20_Training_XXX_t1.nii.gz
  ├── BraTS20_Training_XXX_t1ce.nii.gz
  ├── BraTS20_Training_XXX_t2.nii.gz
  └── BraTS20_Training_XXX_seg.nii.gz
  ```

### 3. Chạy notebook

**Cách 1: Chạy toàn bộ (từ đầu đến cuối)**
- Mở notebook trong Jupyter/VSCode
- Chọn "Run All" hoặc chạy từng cell theo thứ tự

**Cách 2: Chạy từng phần**

#### Phần 1: Preprocessing (Cells 0-10)
- Import libraries
- Define configuration
- Preprocessing functions
- Dataset classes
- Chạy preprocessing để tạo PNG files

#### Phần 2: Model Architecture (Cells 11-13)
- Attention mechanisms (CBAM, Transformer)
- SegUNetV2 model
- BrainTumNetV2 wrapper

#### Phần 3: Loss & Metrics (Cells 14-16)
- Loss functions (Dice, Focal, Boundary)
- Metrics (Dice, IoU, HD95)
- Visualization

#### Phần 4: Training (Cells 17-21)
- Training utilities
- Training loop
- Logger & checkpointing

#### Phần 5: Test & Run (Cells 22-24)
- Test configuration
- Check model architecture
- Start training

### 4. Kiểm tra cấu hình

Trước khi training, chạy cell "Test Configuration & Model Summary" để:
- Kiểm tra config Phase 2
- Verify model architecture
- Test forward pass
- Xem số lượng parameters

### 5. Training

Sau khi preprocessing xong, chạy cell training cuối cùng:
```python
# Trong notebook, điều chỉnh parameters:
cfg = build_phase2_config(
    processed_root=PROCESSED_DIR,
    lmdb_root=LMDB_DIR,
    backend="png",  # hoặc "lmdb"
    model_size="large"  # hoặc "small"
)
cfg["data"]["fold"] = 0  # Chọn fold (0-4)

# Chạy training
best_dice = train_one_fold(cfg, fold=0, config_path=None)
```

### 6. Theo dõi training

**TensorBoard:**
```bash
tensorboard --logdir runs
```

**Log files:**
```
logs/braintumnet_v2_phase2_fold0.log
```

**Checkpoints:**
```
checkpoints/
├── best_fold0.pth  # Best validation Dice
└── last_fold0.pth  # Latest (for resume)
```

## Configuration Details

### Model Config (Phase 2)
```python
{
    "model_type": "v2",
    "base": 64,           # Base channels
    "dim": 512,           # Transformer dimension
    "depth": 4,           # Transformer layers
    "n_heads": 8,         # Attention heads
    "dropout": 0.2,
    "boundary_refinement": True,
    "use_multiscale_transformer": True,  # Phase 2
    "use_attention_gates": True,         # Phase 2
}
```

### Training Config (Phase 2)
```python
{
    "epochs": 400,
    "batch_size": 12,
    "lr": 5e-5,
    "scheduler": "cosine_restarts",  # SGDR
    "T_0": 50,
    "T_mult": 2,
    "boundary_weight": 1.0,          # Phase 2
    "gradient_centralization": True,  # Phase 2
    "focal_alpha": [0.0, 0.35, 0.35, 0.30],
    "class_weights": [1.0, 2.5, 3.0, 4.0],
}
```

## Kết quả mong đợi

- **Target Dice:** 0.93-0.97
- **Improvement over Phase 1:** +2-4% Dice
- **Improvement over Baseline:** +6-11% Dice
- **Training time:** ~8-12 hours/fold (RTX 3090 24GB)

## Troubleshooting

### Out of Memory
- Giảm batch_size: `cfg["train"]["batch_size"] = 8`
- Hoặc dùng model size "small": `model_size="small"` (48/384)

### Data not found
- Kiểm tra đường dẫn trong `DATA_ROOT`
- Đảm bảo đã chạy preprocessing

### CUDA not available
- Model sẽ tự động fallback sang CPU
- Training sẽ chậm hơn rất nhiều

## Liên hệ

Nếu có vấn đề, kiểm tra:
1. Cấu trúc thư mục data
2. Dependencies đã cài đủ chưa
3. GPU memory đủ chưa (recommend 24GB)
4. CUDA version compatibility với PyTorch

## Credits

Model architecture based on:
- SegUNetV2 with Multi-Scale Transformer
- Attention Gates (nnU-Net style)
- Boundary Refinement Module
- BraTS 2020 dataset
