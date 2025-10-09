# Part 6: Utils và Logging

**Navigation**: [[v_TECHNICAL_REPORT_INDEX|← Quay lại Index]]

---

## Mục lục

1. [Tổng quan](#tổng-quan)
2. [I/O Utils (io.py)](#io-utils-iopy)
3. [Training Logger (logger.py)](#training-logger-loggerpy)
4. [Metrics Logger (metrics_logger.py)](#metrics-logger-metrics_loggerpy)
5. [Complete Logging Workflow](#complete-logging-workflow)
6. [Log Analysis](#log-analysis)
7. [Modification Guides](#modification-guides)

---

## Tổng quan

### Mục đích

Các utility modules cung cấp **dịch vụ infrastructure**:
- **I/O**: Load/save models, configs, checkpoints
- **Logging**: Theo dõi tiến trình training, metrics, errors
- **Persistence**: Export CSV/JSON để phân tích

### Các File Chính

| File | Mục đích | Dòng | Hàm Chính |
|------|---------|-------|---------------|
| `utils/io.py` | Checkpoint I/O | 121 | `save_ckpt`, `load_ckpt`, `save_training_state`, `load_training_state` |
| `utils/logger.py` | Text logging | 204 | `TrainingLogger` class |
| `utils/metrics_logger.py` | Metrics export | 124 | `MetricsLogger` class |

---

## I/O Utils (io.py)

**File**: `src/braintumnet/utils/io.py` (121 dòng)

Xử lý tất cả các thao tác file I/O cho project.

### load_yaml

```python
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)
```

**Dòng 4-6**: Load YAML configuration

**Sử dụng**:
```python
cfg = load_yaml("configs/full_dataset_multimodal.yaml")
print(cfg["model"]["base"])  # 32
```

**Tại sao YAML?**
- Configuration dễ đọc
- Hỗ trợ comments
- Cấu trúc phân cấp
- Standard trong các ML projects

---

### ensure_dir

```python
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
```

**Dòng 8-9**: Tạo thư mục nếu chưa tồn tại

**`exist_ok=True`**: Không lỗi nếu thư mục đã tồn tại

**Sử dụng**:
```python
ensure_dir("checkpoints/fold0")
ensure_dir("logs")  # Không lỗi nếu đã tồn tại
```

**Tại sao Cần thiết?**
- Ngăn lỗi "directory not found"
- An toàn cho các processes song song (nhiều folds)

---

### save_ckpt (Chỉ Model)

```python
def save_ckpt(model, path: str):
    """Save only model weights (for best checkpoint)"""
    ensure_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)
```

**Dòng 11-14**: Lưu checkpoint nhẹ

**`state_dict()` là gì?**
- OrderedDict của parameter name → tensor
- Chỉ trainable parameters
- Không có optimizer state, không có scheduler

**Ví dụ**:
```python
model.state_dict() = {
    'seg.e1.block.0.weight': tensor([[...]]),  # Conv weights
    'seg.e1.block.0.bn.weight': tensor([...]),  # BatchNorm gamma
    'seg.e1.block.0.bn.bias': tensor([...]),    # BatchNorm beta
    ...
    'cls_backbone.fc.weight': tensor([[...]]),
    'cls_backbone.fc.bias': tensor([...]),
}
```

**Kích thước File**: ~12 MB cho config mặc định

**Khi nào Sử dụng**:
- Best checkpoint (chỉ để evaluation)
- Deployment models
- Chia sẻ models

---

### load_ckpt (Chỉ Model)

```python
def load_ckpt(model, path: str, map_location="cpu"):
    """Load only model weights"""
    sd = torch.load(path, map_location=map_location)
    model.load_state_dict(sd)
    return model
```

**Dòng 16-20**: Load checkpoint nhẹ

**`map_location`**: Device mapping
```python
# Load GPU checkpoint trên CPU
load_ckpt(model, "checkpoint.pth", map_location="cpu")

# Load CPU checkpoint trên GPU
load_ckpt(model, "checkpoint.pth", map_location="cuda:0")
```

**Tại sao Cần thiết?**
- GPU checkpoints không thể load trên CPU theo mặc định
- Chỉ định target device

**Sử dụng**:
```python
model = BrainTumNet(...)
load_ckpt(model, "checkpoints/best_fold0.pth", map_location="cuda")
model.eval()
# Sẵn sàng cho inference
```

---

### save_training_state (Full State)

```python
def save_training_state(path: str, epoch: int, model, optimizer, scheduler, scaler,
                       best_iou: float, best_iou_epoch: int, config: Dict = None, fold: int = None):
    """
    Save complete training state for resuming.
    """
    ensure_dir(os.path.dirname(path))

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_iou': best_iou,
        'best_iou_epoch': best_iou_epoch,
        'fold': fold,  # Store fold number for validation
    }

    # Add scheduler state if exists
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Add scaler state if exists
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    # Add config if provided
    if config is not None:
        checkpoint['config'] = config

    torch.save(checkpoint, path)
    print(f"Saved training state to: {path}")
```

**Dòng 22-62**: Lưu complete training state

**Những gì được Lưu**:

1. **Thông tin Cơ bản**:
```python
{
    'epoch': 49,  # Epoch hiện tại (0-indexed)
    'best_iou': 0.8430,
    'best_iou_epoch': 44,
    'fold': 0,
}
```

2. **Model Weights**:
```python
'model_state_dict': model.state_dict()
# Giống save_ckpt - tất cả model parameters
```

3. **Optimizer State**:
```python
'optimizer_state_dict': optimizer.state_dict()
```

**Optimizer state chứa gì?**
```python
optimizer.state_dict() = {
    'state': {
        0: {  # Cho parameter đầu tiên
            'step': 50000,  # Số lần updates
            'exp_avg': tensor([...]),  # First moment (momentum)
            'exp_avg_sq': tensor([...]),  # Second moment (RMSprop)
        },
        1: {  # Cho parameter thứ hai
            'step': 50000,
            'exp_avg': tensor([...]),
            'exp_avg_sq': tensor([...]),
        },
        ...
    },
    'param_groups': [{
        'lr': 5e-5,  # Learning rate hiện tại
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 1e-5,
        ...
    }]
}
```

**Tại sao Lưu Điều này?**
- Adam duy trì running averages (momentum)
- Resume mà không có điều này = restart optimization từ đầu
- Giảm hiệu suất: 5-10 epochs để phục hồi

4. **Scheduler State** (nếu tồn tại):
```python
if scheduler is not None:
    checkpoint['scheduler_state_dict'] = scheduler.state_dict()
```

**ReduceLROnPlateau state**:
```python
scheduler.state_dict() = {
    'best': 0.8430,  # Metric tốt nhất đã thấy
    'num_bad_epochs': 5,  # Epochs không cải thiện
    'cooldown_counter': 0,
    'mode': 'max',
    'patience': 10,
    ...
}
```

**Tại sao Lưu Điều này?**
- Scheduler theo dõi patience counter
- Không có điều này: LR có thể giảm quá sớm/muộn

5. **Scaler State** (nếu sử dụng mixed precision):
```python
if scaler is not None:
    checkpoint['scaler_state_dict'] = scaler.state_dict()
```

**GradScaler state**:
```python
scaler.state_dict() = {
    'scale': 65536.0,  # Loss scale hiện tại
    'growth_tracker': 0,
    'growth_factor': 2.0,
    'backoff_factor': 0.5,
    'growth_interval': 2000,
    ...
}
```

**Tại sao Lưu Điều này?**
- Scale factor thích ứng trong training
- Không có điều này: Có thể overflow/underflow sau khi resume

6. **Config** (tùy chọn):
```python
if config is not None:
    checkpoint['config'] = config
```

**Toàn bộ YAML config được lưu**:
- Đảm bảo reproducibility
- Có thể xác minh settings khi resuming

**Kích thước File**: ~25 MB (so với ~12 MB cho model-only)

**Phân tích Bộ nhớ**:
- Model: ~12 MB
- Optimizer: ~12 MB (2× momentum buffers)
- Scheduler: <1 KB
- Scaler: <1 KB
- Config: <1 KB

---

### load_training_state (Full State)

```python
def load_training_state(path: str, model, optimizer, scheduler=None, scaler=None, map_location="cpu", expected_fold=None):
    """
    Load complete training state for resuming.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location)

    # Validate fold if provided
    checkpoint_fold = checkpoint.get('fold', None)
    if expected_fold is not None and checkpoint_fold is not None:
        if checkpoint_fold != expected_fold:
            raise ValueError(
                f"Fold mismatch! Checkpoint is for fold {checkpoint_fold}, "
                f"but you're trying to resume fold {expected_fold}. "
                f"Please use the correct checkpoint: last_fold{expected_fold}.pth"
            )
```

**Dòng 64-92**: Load và validate checkpoint

**Fold Validation** (QUAN TRỌNG):
```python
if checkpoint_fold != expected_fold:
    raise ValueError(...)
```

**Tại sao Điều này Quan trọng**:

**Tình huống Không có Validation**:
```bash
# Train fold 0 đến epoch 50
python train.py --fold 0

# Vô tình resume với fold 1 checkpoint
python train.py --fold 0 --resume checkpoints/last_fold1.pth

# XẤU: Training tiếp tục với:
#   - Fold 0 data
#   - Fold 1 model weights
#   - Kết quả vô nghĩa!
```

**Tình huống Có Validation**:
```bash
python train.py --fold 0 --resume checkpoints/last_fold1.pth

# LỖI:
# ValueError: Fold mismatch! Checkpoint is for fold 1,
# but you're trying to resume fold 0.
# Please use the correct checkpoint: last_fold0.pth

# User sửa lệnh:
python train.py --fold 0 --resume checkpoints/last_fold0.pth
# THÀNH CÔNG ✓
```

---

```python
    # Load model
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Load scaler if provided
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
```

**Dòng 94-106**: Khôi phục tất cả states

**In-Place Loading**:
- `model.load_state_dict()`: Cập nhật model parameters
- `optimizer.load_state_dict()`: Cập nhật momentum buffers
- Các objects được modified in-place (không thay thế)

---

```python
    # Return training info
    info = {
        'epoch': checkpoint['epoch'],
        'best_iou': checkpoint.get('best_iou', -1.0),
        'best_iou_epoch': checkpoint.get('best_iou_epoch', 0),
        'config': checkpoint.get('config', None),
    }

    print(f"Loaded training state from: {path}")
    print(f"  Resuming from epoch {info['epoch'] + 1}")
    print(f"  Best IoU so far: {info['best_iou']:.4f} (epoch {info['best_iou_epoch'] + 1})")

    return info
```

**Dòng 108-120**: Trả về training metadata

**Sử dụng trong Trainer**:
```python
resume_info = load_training_state(
    resume_from, model, opt, plateau_scheduler, scaler, device, expected_fold=fold
)

start_epoch = resume_info['epoch'] + 1  # Epoch tiếp theo
best_iou = resume_info['best_iou']
best_iou_epoch = resume_info['best_iou_epoch']

print(f"Resuming from epoch {start_epoch}")
# Training tiếp tục mượt mà
```

---

## Training Logger (logger.py)

**File**: `src/braintumnet/utils/logger.py` (204 dòng)

Cung cấp **text logging dễ đọc** ra file và console.

### Initialization

```python
class TrainingLogger:
    def __init__(self, log_dir, exp_name, fold, console=True):
        self.console = console
        self.exp_name = exp_name
        self.fold = fold

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{exp_name}_fold{fold}_{timestamp}.log")

        # Initialize log file
        self.start_time = datetime.now()
        self._write_header()
```

**Dòng 18-39**: Khởi tạo logger

**Định dạng Tên File Log**:
```
{exp_name}_fold{fold}_{timestamp}.log

Ví dụ:
multimodal_training_fold0_20240115_103045.log
```

**Tại sao Timestamp?**
- Nhiều training runs không ghi đè
- Dễ theo dõi lịch sử experiments
- Có thể so sánh các runs khác nhau

---

### Header

```python
    def _write_header(self):
        """Write log file header."""
        with open(self.log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BrainTumNet Training Log\n")
            f.write("=" * 80 + "\n")
            f.write(f"Experiment: {self.exp_name}\n")
            f.write(f"Fold: {self.fold}\n")
            f.write(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 80 + "\n\n")
```

**Dòng 41-50**: Viết header

**Ví dụ Output**:
```
================================================================================
BrainTumNet Training Log
================================================================================
Experiment: multimodal_training
Fold: 0
Start Time: 2024-01-15 10:30:45
--------------------------------------------------------------------------------

```

---

### Logging Methods

```python
    def log(self, message, level="INFO"):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] [{level}] {message}"

        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(formatted_msg + "\n")

        # Print to console
        if self.console:
            print(formatted_msg)

    def info(self, message):
        """Log info message."""
        self.log(message, "INFO")

    def warning(self, message):
        """Log warning message."""
        self.log(message, "WARNING")

    def error(self, message):
        """Log error message."""
        self.log(message, "ERROR")

    def success(self, message):
        """Log success message."""
        self.log(message, "SUCCESS")
```

**Dòng 52-85**: Các phương thức logging

**Log Levels**:
- **INFO**: Thông tin chung (dataset loaded, epoch started)
- **WARNING**: Vấn đề không nghiêm trọng (NaN detected, slow training)
- **ERROR**: Lỗi nghiêm trọng (file not found, OOM)
- **SUCCESS**: Milestones quan trọng (new best checkpoint)

**Sử dụng**:
```python
logger = TrainingLogger("logs", "exp1", fold=0)

logger.info("Loading dataset")
# [10:30:50] [INFO] Loading dataset

logger.warning("Learning rate reduced")
# [11:45:23] [WARNING] Learning rate reduced

logger.error("Checkpoint not found")
# [12:00:00] [ERROR] Checkpoint not found

logger.success("New best IoU achieved!")
# [13:30:15] [SUCCESS] New best IoU achieved!
```

---

### Section Headers

```python
    def section(self, title):
        """Log a section header."""
        with open(self.log_file, 'a') as f:
            f.write("\n" + "-" * 80 + "\n")
            f.write(f"{title}\n")
            f.write("-" * 80 + "\n")

        if self.console:
            print("\n" + "-" * 80)
            print(f"{title}")
            print("-" * 80)
```

**Dòng 87-97**: Section separators

**Output**:
```
--------------------------------------------------------------------------------
Epoch 1/100 - TRAIN
--------------------------------------------------------------------------------
```

**Mục đích**: Tổ chức trực quan trong các log files dài

---

### Epoch Logging

```python
    def epoch_start(self, epoch, total_epochs, phase="TRAIN"):
        """Log epoch start."""
        msg = f"Epoch {epoch+1}/{total_epochs} - {phase}"
        self.section(msg)

    def epoch_end(self, epoch, total_epochs, metrics, phase="TRAIN"):
        """Log epoch end with metrics."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        msg = f"[{timestamp}] Epoch {epoch+1}/{total_epochs} - {phase} - {metrics_str}"

        with open(self.log_file, 'a') as f:
            f.write(msg + "\n")

        if self.console:
            print(msg)
```

**Dòng 99-122**: Epoch logging

**Ví dụ**:
```python
logger.epoch_start(0, 100, "TRAIN")
# --------------------------------------------------------------------------------
# Epoch 1/100 - TRAIN
# --------------------------------------------------------------------------------

metrics = {
    'train_loss': 0.8234,
    'val_iou': 0.4523,
    'val_dice': 0.6234,
    'lr': 0.0001
}
logger.epoch_end(0, 100, metrics, "SUMMARY")
# [10:35:23] Epoch 1/100 - SUMMARY - train_loss: 0.8234, val_iou: 0.4523, val_dice: 0.6234, lr: 0.0001
```

---

### Best Checkpoint

```python
    def best_checkpoint(self, metric_name, metric_value, epoch):
        """Log new best checkpoint."""
        msg = f"*** NEW BEST {metric_name.upper()}: {metric_value:.4f} (epoch {epoch+1}) - Checkpoint saved ***"
        self.success(msg)
```

**Dòng 124-127**: Thông báo best checkpoint

**Ví dụ**:
```python
logger.best_checkpoint("IoU", 0.8430, 54)
# [13:45:12] [SUCCESS] *** NEW BEST IOU: 0.8430 (epoch 55) - Checkpoint saved ***
```

**Nổi bật**: Dễ tìm best checkpoints trong log

---

### Config Saving

```python
    def save_config(self, config, config_path):
        """Save configuration to log directory."""
        import shutil
        import yaml

        # Copy original config
        log_dir = os.path.dirname(self.log_file)
        config_copy = os.path.join(log_dir, f"config_fold{self.fold}.yaml")
        shutil.copy(config_path, config_copy)

        # Also save as JSON for easy parsing
        config_json = os.path.join(log_dir, f"config_fold{self.fold}.json")
        with open(config_json, 'w') as f:
            json.dump(config, f, indent=2)

        self.info(f"Configuration saved to: {config_copy}")
        self.info(f"Configuration (JSON) saved to: {config_json}")
```

**Dòng 129-151**: Lưu config

**Tại sao Lưu Config?**
- Reproducibility: Biết chính xác settings đã sử dụng
- So sánh: Diff configs giữa các experiments
- Documentation: Bản ghi experiments độc lập

**Hai Định dạng**:
- **YAML**: Dễ đọc, có comments
- **JSON**: Dễ parse bằng máy

---

### Training Summary

```python
    def training_summary(self, best_metrics, total_time):
        """Log training summary."""
        with open(self.log_file, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("Training Complete!\n")
            f.write("=" * 80 + "\n")

            # Format time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            f.write(f"Total Time: {hours}h {minutes}m {seconds}s\n")

            # Best metrics
            f.write("\nBest Metrics:\n")
            for metric, value in best_metrics.items():
                if isinstance(value, tuple):
                    val, ep = value
                    f.write(f"  {metric}: {val:.4f} (epoch {ep+1})\n")
                else:
                    f.write(f"  {metric}: {value:.4f}\n")

            f.write("\nLog file: " + self.log_file + "\n")
            f.write("=" * 80 + "\n")
```

**Dòng 153-197**: Tóm tắt cuối cùng

**Ví dụ Output**:
```
================================================================================
Training Complete!
================================================================================
Total Time: 2h 15m 38s

Best Metrics:
  iou: 0.8430 (epoch 55)
  dice: 0.9148 (epoch 55)
  acc: 0.9823 (epoch 60)

Log file: logs/multimodal_training_fold0_20240115_103045.log
================================================================================
```

---

## Metrics Logger (metrics_logger.py)

**File**: `src/braintumnet/utils/metrics_logger.py` (124 dòng)

Cung cấp **structured metrics export** ra CSV và JSON.

### Initialization

```python
class MetricsLogger:
    def __init__(self, log_dir, exp_name, fold):
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.fold = fold

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Metric storage
        self.metrics_history = []
        self.best_metrics = {}

        # File paths
        self.csv_path = os.path.join(log_dir, f"metrics_{exp_name}_fold{fold}.csv")
        self.json_path = os.path.join(log_dir, f"metrics_{exp_name}_fold{fold}.json")

        # CSV file initialization
        self.csv_initialized = False
        self.csv_headers = None
```

**Dòng 17-41**: Khởi tạo

**Cấu trúc Dữ liệu**:
```python
self.metrics_history = [
    {'epoch': 0, 'train_loss': 0.82, 'val_iou': 0.45, ...},
    {'epoch': 1, 'train_loss': 0.67, 'val_iou': 0.57, ...},
    ...
]

self.best_metrics = {
    'val_iou': (0.8430, 54),  # (value, epoch)
    'val_dice': (0.9148, 54),
    'train_loss': (0.0823, 98),  # Thấp hơn là tốt hơn
    ...
}
```

---

### Log Epoch

```python
    def log_epoch(self, epoch, metrics_dict):
        """Log metrics for one epoch."""
        # Add epoch number
        metrics_dict['epoch'] = epoch

        # Store in history
        self.metrics_history.append(metrics_dict.copy())

        # Update best metrics
        for key, value in metrics_dict.items():
            if key == 'epoch':
                continue

            # For loss, lower is better; for others, higher is better
            if 'loss' in key.lower():
                if key not in self.best_metrics or value < self.best_metrics[key][0]:
                    self.best_metrics[key] = (value, epoch)
            else:
                if key not in self.best_metrics or value > self.best_metrics[key][0]:
                    self.best_metrics[key] = (value, epoch)

        # Write to CSV
        self._write_csv(metrics_dict)
```

**Dòng 43-71**: Log một epoch

**Theo dõi Best Tự động**:
```python
# Loss: thấp hơn là tốt hơn
if 'loss' in key.lower():
    if value < best:
        best_metrics[key] = (value, epoch)

# Khác (IoU, Dice, Acc): cao hơn là tốt hơn
else:
    if value > best:
        best_metrics[key] = (value, epoch)
```

**Ví dụ**:
```python
logger = MetricsLogger("logs", "exp1", 0)

# Epoch 0
logger.log_epoch(0, {
    'train_loss': 0.82,
    'val_iou': 0.45,
    'val_dice': 0.62
})
# best_metrics = {'train_loss': (0.82, 0), 'val_iou': (0.45, 0), 'val_dice': (0.62, 0)}

# Epoch 1
logger.log_epoch(1, {
    'train_loss': 0.67,  # Tốt hơn (thấp hơn)
    'val_iou': 0.57,     # Tốt hơn (cao hơn)
    'val_dice': 0.72     # Tốt hơn (cao hơn)
})
# best_metrics = {'train_loss': (0.67, 1), 'val_iou': (0.57, 1), 'val_dice': (0.72, 1)}

# Epoch 2
logger.log_epoch(2, {
    'train_loss': 0.70,  # Kém hơn - không cập nhật
    'val_iou': 0.65,     # Tốt hơn - cập nhật
    'val_dice': 0.70     # Kém hơn - không cập nhật
})
# best_metrics = {'train_loss': (0.67, 1), 'val_iou': (0.65, 2), 'val_dice': (0.72, 1)}
```

---

### CSV Writing

```python
    def _write_csv(self, metrics_dict):
        """Write metrics to CSV file."""
        # Initialize CSV headers on first write
        if not self.csv_initialized:
            self.csv_headers = sorted(metrics_dict.keys())
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_headers)
                writer.writeheader()
            self.csv_initialized = True

        # Append metrics
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            writer.writerow(metrics_dict)
```

**Dòng 73-86**: Viết ra CSV

**Incremental Writing**:
- Mỗi epoch được append ngay lập tức
- Không mất dữ liệu nếu training crashes
- Có thể monitor progress trong khi training

**Định dạng CSV**:
```csv
epoch,learning_rate,train_loss,val_acc,val_dice,val_iou
0,0.0001,0.8234,0.6789,0.6234,0.4523
1,0.0001,0.6745,0.7234,0.7234,0.5678
2,0.0001,0.5432,0.7456,0.7589,0.6123
...
```

**Load trong Pandas**:
```python
import pandas as pd
df = pd.read_csv("logs/metrics_exp1_fold0.csv")
print(df.head())

# Vẽ metrics
import matplotlib.pyplot as plt
plt.plot(df['epoch'], df['val_dice'], label='Dice')
plt.plot(df['epoch'], df['val_iou'], label='IoU')
plt.legend()
plt.show()
```

---

### JSON Export

```python
    def save_json(self):
        """Save all metrics to JSON file."""
        output = {
            'experiment': self.exp_name,
            'fold': self.fold,
            'history': self.metrics_history,
            'best_metrics': {k: {'value': v[0], 'epoch': v[1]}
                            for k, v in self.best_metrics.items()}
        }

        with open(self.json_path, 'w') as f:
            json.dump(output, f, indent=2)
```

**Dòng 88-99**: Lưu JSON

**Định dạng JSON**:
```json
{
  "experiment": "multimodal_training",
  "fold": 0,
  "history": [
    {
      "epoch": 0,
      "train_loss": 0.8234,
      "val_iou": 0.4523,
      "val_dice": 0.6234,
      "val_acc": 0.6789,
      "learning_rate": 0.0001
    },
    {
      "epoch": 1,
      "train_loss": 0.6745,
      "val_iou": 0.5678,
      "val_dice": 0.7234,
      "val_acc": 0.7234,
      "learning_rate": 0.0001
    }
  ],
  "best_metrics": {
    "train_loss": {"value": 0.1234, "epoch": 98},
    "val_iou": {"value": 0.8430, "epoch": 54},
    "val_dice": {"value": 0.9148, "epoch": 54},
    "val_acc": {"value": 0.9823, "epoch": 60}
  }
}
```

**Load trong Python**:
```python
import json
with open("logs/metrics_exp1_fold0.json") as f:
    data = json.load(f)

print(f"Best Dice: {data['best_metrics']['val_dice']['value']:.4f} "
      f"at epoch {data['best_metrics']['val_dice']['epoch']}")
# Best Dice: 0.9148 at epoch 54
```

---

### Print Summary

```python
    def print_summary(self):
        """Print summary of best metrics."""
        print("\n" + "=" * 60)
        print("Best Metrics Summary")
        print("=" * 60)
        for metric, (value, epoch) in sorted(self.best_metrics.items()):
            print(f"{metric:20s}: {value:.4f} (epoch {epoch+1})")
        print("=" * 60)
        print(f"CSV saved to: {self.csv_path}")
        print(f"JSON saved to: {self.json_path}")
        print("=" * 60 + "\n")
```

**Dòng 113-123**: In tóm tắt

**Ví dụ Output**:
```
============================================================
Best Metrics Summary
============================================================
learning_rate       : 0.0001 (epoch 1)
train_loss          : 0.1234 (epoch 99)
val_acc             : 0.9823 (epoch 61)
val_dice            : 0.9148 (epoch 55)
val_iou             : 0.8430 (epoch 55)
============================================================
CSV saved to: logs/metrics_multimodal_training_fold0.csv
JSON saved to: logs/metrics_multimodal_training_fold0.json
============================================================
```

---

## Complete Logging Workflow

### Training Integration

```python
# Trong trainer.py
from braintumnet.utils.logger import TrainingLogger
from braintumnet.utils.metrics_logger import MetricsLogger

def train_one_fold(cfg, fold):
    # Khởi tạo loggers
    logger = TrainingLogger("logs", cfg["exp_name"], fold)
    metrics_logger = MetricsLogger("logs", cfg["exp_name"], fold)

    logger.info("Starting training...")

    for epoch in range(epochs):
        # Training
        logger.epoch_start(epoch, epochs, "TRAIN")
        train_loss = train_epoch(...)

        # Validation
        val_iou, val_dice, val_acc = validate(...)

        # Log ra text logger
        logger.epoch_end(epoch, epochs, {
            'train_loss': train_loss,
            'val_iou': val_iou,
            'val_dice': val_dice,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        }, "SUMMARY")

        # Log ra metrics logger (CSV/JSON)
        metrics_logger.log_epoch(epoch, {
            'train_loss': train_loss,
            'val_iou': val_iou,
            'val_dice': val_dice,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        # Kiểm tra best checkpoint
        if val_iou > best_iou:
            best_iou = val_iou
            logger.best_checkpoint("IoU", val_iou, epoch)

    # Training hoàn thành
    best_metrics = metrics_logger.get_best_metrics()
    logger.training_summary(best_metrics, total_time)
    metrics_logger.print_summary()
    metrics_logger.close()
```

---

### Output Files

Sau khi training, bạn sẽ có:

```
logs/
├── multimodal_training_fold0_20240115_103045.log    # Text log
├── config_fold0.yaml                                  # Config copy
├── config_fold0.json                                  # Config (JSON)
├── metrics_multimodal_training_fold0.csv             # Metrics CSV
└── metrics_multimodal_training_fold0.json            # Metrics JSON
```

---

## Log Analysis

### Pandas Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
df = pd.read_csv("logs/metrics_multimodal_training_fold0.csv")

# Vẽ training curve
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Loss
axes[0, 0].plot(df['epoch'], df['train_loss'])
axes[0, 0].set_title('Training Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True)

# IoU
axes[0, 1].plot(df['epoch'], df['val_iou'])
axes[0, 1].set_title('Validation IoU')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('IoU')
axes[0, 1].grid(True)

# Dice
axes[1, 0].plot(df['epoch'], df['val_dice'])
axes[1, 0].set_title('Validation Dice')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Dice')
axes[1, 0].grid(True)

# Learning Rate
axes[1, 1].plot(df['epoch'], df['learning_rate'])
axes[1, 1].set_title('Learning Rate')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('LR')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()

# Tìm best epoch
best_epoch = df.loc[df['val_dice'].idxmax()]
print(f"Best Dice: {best_epoch['val_dice']:.4f} at epoch {best_epoch['epoch']}")
```

---

### Compare Folds

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load tất cả folds
folds = []
for fold in range(5):
    df = pd.read_csv(f"logs/metrics_multimodal_training_fold{fold}.csv")
    df['fold'] = fold
    folds.append(df)

all_data = pd.concat(folds)

# Vẽ tất cả folds
plt.figure(figsize=(10, 6))
for fold in range(5):
    fold_data = all_data[all_data['fold'] == fold]
    plt.plot(fold_data['epoch'], fold_data['val_dice'], label=f'Fold {fold}', alpha=0.7)

plt.xlabel('Epoch')
plt.ylabel('Dice Score')
plt.title('Validation Dice Across All Folds')
plt.legend()
plt.grid(True)
plt.savefig("all_folds_comparison.png", dpi=150)
plt.show()

# Thống kê tổng hợp
final_dice = all_data.groupby('fold')['val_dice'].max()
print(f"Dice scores per fold: {final_dice.values}")
print(f"Mean ± Std: {final_dice.mean():.4f} ± {final_dice.std():.4f}")
```

---

## Modification Guides

### Thêm Custom Metric vào Logger

```python
# Trong metrics_logger.py, sửa log_epoch method
def log_epoch(self, epoch, metrics_dict):
    # ... existing code ...

    # Custom best metric logic cho precision/recall balance
    if 'precision' in metrics_dict and 'recall' in metrics_dict:
        f1 = 2 * (metrics_dict['precision'] * metrics_dict['recall']) / \
             (metrics_dict['precision'] + metrics_dict['recall'] + 1e-6)
        metrics_dict['f1_score'] = f1

    # ... rest of code ...
```

### Thêm Email Notifications

```python
# Trong logger.py
import smtplib
from email.mime.text import MIMEText

class TrainingLogger:
    def __init__(self, log_dir, exp_name, fold, console=True, email=None):
        # ... existing code ...
        self.email = email

    def send_email(self, subject, message):
        """Send email notification."""
        if self.email is None:
            return

        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = 'training@braintumnet.com'
        msg['To'] = self.email

        with smtplib.SMTP('localhost') as server:
            server.send_message(msg)

    def best_checkpoint(self, metric_name, metric_value, epoch):
        """Log new best checkpoint."""
        msg = f"*** NEW BEST {metric_name.upper()}: {metric_value:.4f} (epoch {epoch+1}) ***"
        self.success(msg)

        # Gửi email notification
        email_msg = f"New best {metric_name}: {metric_value:.4f} at epoch {epoch+1}\n"
        email_msg += f"Experiment: {self.exp_name}, Fold: {self.fold}"
        self.send_email(f"BrainTumNet: New Best {metric_name}", email_msg)
```

### Thêm Slack Notifications

```python
# Trong logger.py
import requests

class TrainingLogger:
    def __init__(self, log_dir, exp_name, fold, console=True, slack_webhook=None):
        # ... existing code ...
        self.slack_webhook = slack_webhook

    def send_slack(self, message):
        """Send Slack notification."""
        if self.slack_webhook is None:
            return

        payload = {'text': message}
        requests.post(self.slack_webhook, json=payload)

    def training_summary(self, best_metrics, total_time):
        """Log training summary."""
        # ... existing code ...

        # Gửi Slack notification
        msg = f":white_check_mark: Training Complete: {self.exp_name} Fold {self.fold}\n"
        msg += f"Best IoU: {best_metrics['iou'][0]:.4f}\n"
        msg += f"Best Dice: {best_metrics['dice'][0]:.4f}\n"
        msg += f"Total Time: {hours}h {minutes}m"
        self.send_slack(msg)
```

---

**Tiếp theo**: [[v_07_CONFIGURATION_SYSTEM|Part 7: Configuration System →]]

**Quay lại**: [[v_05_EVALUATION_INFERENCE|← Part 5: Evaluation and Inference]] | [[v_TECHNICAL_REPORT_INDEX|Index]]
