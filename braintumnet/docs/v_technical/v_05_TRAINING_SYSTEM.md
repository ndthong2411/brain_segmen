# Phần 5: Training System Chi Tiết

> **🏋️ Hệ Thống Training - Trainer, Optimizer, Scheduler, Mixed Precision**
>
> Tài liệu này giải thích chi tiết training loop, optimization strategies, và monitoring.

---

## Mục Lục

1. [Tổng Quan Training Pipeline](#1-tổng-quan-training-pipeline)
2. [Trainer Class](#2-trainer-class)
3. [Optimizer Configuration](#3-optimizer-configuration)
4. [Learning Rate Scheduler](#4-learning-rate-scheduler)
5. [Mixed Precision Training](#5-mixed-precision-training)
6. [Gradient Accumulation](#6-gradient-accumulation)
7. [Checkpointing](#7-checkpointing)
8. [Monitoring và Logging](#8-monitoring-và-logging)

---

## 1. Tổng Quan Training Pipeline

### Training Flow

```
Epoch Loop (250 epochs):
    │
    ├─→ Training Phase:
    │   │
    │   └─→ Batch Loop:
    │       ├─ Load batch (images, masks, labels)
    │       ├─ Forward pass
    │       ├─ Compute losses (seg + cls + deep supervision)
    │       ├─ Backward pass (mixed precision)
    │       ├─ Optimizer step
    │       ├─ Update metrics
    │       └─ Log to TensorBoard
    │
    ├─→ Validation Phase:
    │   │
    │   └─→ Batch Loop:
    │       ├─ Forward pass (no grad)
    │       ├─ Compute metrics (Dice WT/TC/ED)
    │       └─ Accumulate results
    │
    ├─→ Scheduler Step (update learning rate)
    │
    ├─→ Save Checkpoint (best model)
    │
    └─→ Log Metrics (CSV, TensorBoard, console)
```

### File Code

**File**: `src/braintumnet/engine/trainer.py` (485 dòng)

---

## 2. Trainer Class

### Initialization

```python
class Trainer:
    """
    Training engine cho BrainTumNet
    
    Features:
    - Multi-task learning (segmentation + classification)
    - Deep supervision support
    - Mixed precision (AMP)
    - Gradient accumulation
    - Comprehensive logging
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        seg_loss_fn,
        cls_loss_fn,
        device,
        num_epochs=250,
        gradient_accumulation_steps=1,
        use_amp=True,
        amp_dtype='float16',
        checkpoint_dir='checkpoints',
        log_dir='logs',
        tensorboard_dir='runs',
        deep_supervision=True,
        aux_weights=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.seg_loss_fn = seg_loss_fn
        self.cls_loss_fn = cls_loss_fn
        self.device = device
        
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp
        self.deep_supervision = deep_supervision
        
        # Mixed precision scaler
        if use_amp:
            if amp_dtype == 'float16':
                self.scaler = torch.cuda.amp.GradScaler()
            elif amp_dtype == 'bfloat16':
                self.scaler = None  # bfloat16 không cần scaler
        else:
            self.scaler = None
        
        # Directories
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(tensorboard_dir)
        
        # Metrics tracking
        self.best_val_dice = 0.0
        self.epoch = 0
        
        # Auxiliary weights cho deep supervision
        if aux_weights is None:
            self.aux_weights = [0.2, 0.3, 0.5]
        else:
            self.aux_weights = aux_weights
```

### Training Epoch

```python
def train_epoch(self):
    """
    Một epoch training
    
    Returns:
        metrics: Dict với training metrics
    """
    self.model.train()
    
    # Accumulators
    total_loss = 0.0
    total_seg_loss = 0.0
    total_cls_loss = 0.0
    num_batches = 0
    
    # Progress bar
    pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch+1}/{self.num_epochs}')
    
    for batch_idx, (images, masks, labels) in enumerate(pbar):
        # Move to device
        images = images.to(self.device)
        masks = masks.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass với mixed precision
        with torch.cuda.amp.autocast(
            enabled=self.use_amp,
            dtype=torch.float16 if self.amp_dtype == 'float16' else torch.bfloat16
        ):
            # Model forward
            if self.deep_supervision:
                seg_logits, cls_logits, aux_outputs = self.model(images)
            else:
                seg_logits, cls_logits = self.model(images)
                aux_outputs = None
            
            # Segmentation loss
            if self.deep_supervision and aux_outputs is not None:
                seg_loss = self._compute_deep_supervision_loss(
                    seg_logits, aux_outputs, masks
                )
            else:
                seg_loss, _ = self.seg_loss_fn(seg_logits, masks)
            
            # Classification loss
            cls_loss = self.cls_loss_fn(cls_logits, labels)
            
            # Total loss
            loss = seg_loss + cls_loss
            
            # Normalize by accumulation steps
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step (after accumulation)
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            if self.use_amp and self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
        
        # Update metrics
        total_loss += loss.item() * self.gradient_accumulation_steps
        total_seg_loss += seg_loss.item()
        total_cls_loss += cls_loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'seg': f'{seg_loss.item():.4f}',
            'cls': f'{cls_loss.item():.4f}'
        })
    
    # Average metrics
    metrics = {
        'train_loss': total_loss / num_batches,
        'train_seg_loss': total_seg_loss / num_batches,
        'train_cls_loss': total_cls_loss / num_batches
    }
    
    return metrics
```

### Validation Epoch

```python
def validate_epoch(self):
    """
    Validation epoch
    
    Returns:
        metrics: Dict với validation metrics
    """
    self.model.eval()
    
    # Metrics accumulator
    accumulator = MulticlassMetricsAccumulator(
        num_classes=3, ignore_background=True
    )
    
    total_val_loss = 0.0
    total_seg_loss = 0.0
    total_cls_loss = 0.0
    num_batches = 0
    
    cls_correct = 0
    cls_total = 0
    
    with torch.no_grad():
        for images, masks, labels in tqdm(self.val_loader, desc='Validation'):
            # Move to device
            images = images.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            if self.deep_supervision:
                seg_logits, cls_logits, aux_outputs = self.model(images)
            else:
                seg_logits, cls_logits = self.model(images)
            
            # Losses
            seg_loss, _ = self.seg_loss_fn(seg_logits, masks)
            cls_loss = self.cls_loss_fn(cls_logits, labels)
            
            total_seg_loss += seg_loss.item()
            total_cls_loss += cls_loss.item()
            total_val_loss += (seg_loss + cls_loss).item()
            num_batches += 1
            
            # Segmentation predictions
            seg_pred = seg_logits.argmax(dim=1)  # (B, H, W)
            
            # Update metrics accumulator
            accumulator.update(seg_pred, masks)
            
            # Classification accuracy
            cls_pred = cls_logits.argmax(dim=1)
            cls_correct += (cls_pred == labels).sum().item()
            cls_total += labels.size(0)
    
    # Compute final metrics
    seg_metrics = accumulator.compute()
    cls_accuracy = cls_correct / cls_total
    
    # Combine metrics
    metrics = {
        'val_loss': total_val_loss / num_batches,
        'val_seg_loss': total_seg_loss / num_batches,
        'val_cls_loss': total_cls_loss / num_batches,
        'val_dice_wt': seg_metrics['WT'],
        'val_dice_tc': seg_metrics['TC'],
        'val_dice_ed': seg_metrics['ED'],
        'val_dice_mean': seg_metrics['Mean'],
        'val_cls_acc': cls_accuracy
    }
    
    return metrics
```

### Main Training Loop

```python
def train(self):
    """
    Main training loop
    """
    print(f"Starting training for {self.num_epochs} epochs...")
    print(f"Device: {self.device}")
    print(f"Mixed Precision: {self.use_amp}")
    print(f"Gradient Accumulation: {self.gradient_accumulation_steps}")
    
    for epoch in range(self.num_epochs):
        self.epoch = epoch
        
        # Train
        train_metrics = self.train_epoch()
        
        # Validate
        val_metrics = self.validate_epoch()
        
        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Log metrics
        self._log_metrics(train_metrics, val_metrics)
        
        # Save checkpoint
        self._save_checkpoint(val_metrics)
        
        # Print summary
        print(f"\n[Epoch {epoch+1}/{self.num_epochs}]")
        print(f"Train Loss: {train_metrics['train_loss']:.4f} | "
              f"Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"WT: {val_metrics['val_dice_wt']:.4f} | "
              f"TC: {val_metrics['val_dice_tc']:.4f} | "
              f"ED: {val_metrics['val_dice_ed']:.4f} | "
              f"Mean: {val_metrics['val_dice_mean']:.4f}")
        print(f"Cls Acc: {val_metrics['val_cls_acc']:.4f}")
        
    print("\n✓ Training completed!")
    self.writer.close()
```

---

## 3. Optimizer Configuration

### AdamW Optimizer

```python
def create_optimizer(model, cfg):
    """
    Create AdamW optimizer với weight decay
    
    Args:
        model: BrainTumNet model
        cfg: Config dict
    
    Returns:
        optimizer: torch.optim.AdamW
    """
    # Separate parameters: với và không có weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # No decay for bias và normalization layers
        if 'bias' in name or 'norm' in name or 'bn' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    # Parameter groups
    param_groups = [
        {'params': decay_params, 'weight_decay': cfg['weight_decay']},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    # AdamW optimizer
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg['learning_rate'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    return optimizer
```

**Tại sao separate weight decay?**
```
Weight decay = L2 regularization
- Good cho weights (prevent overfitting)
- Bad cho biases (already constrained)
- Bad cho normalization parameters (interferes với statistics)

→ Apply weight decay chỉ cho weight matrices
```

### Hyperparameters

```python
# Typical configuration
optimizer_cfg = {
    'learning_rate': 1e-4,        # Base LR
    'weight_decay': 1e-5,         # L2 regularization
    'betas': (0.9, 0.999),        # Adam momentum
    'eps': 1e-8                   # Numerical stability
}

# A100 optimized (higher LR với large batch)
a100_cfg = {
    'learning_rate': 3e-4,
    'weight_decay': 1e-5,
}

# RTX 3090 (smaller batch → lower LR)
rtx3090_cfg = {
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
}
```

---

## 4. Learning Rate Scheduler

### Cosine Annealing với Warmup

```python
def create_scheduler(optimizer, cfg):
    """
    Create learning rate scheduler
    
    Strategy: Warmup + Cosine Annealing
    
    Epochs 0-10:   Linear warmup (0 → base_lr)
    Epochs 10-250: Cosine decay (base_lr → min_lr)
    """
    num_epochs = cfg['num_epochs']
    warmup_epochs = cfg.get('warmup_epochs', 10)
    min_lr = cfg.get('min_lr', 1e-6)
    
    # Lambda function cho warmup + cosine
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return epoch / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return min_lr + (1 - min_lr) * cosine_decay
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_lambda
    )
    
    return scheduler
```

**Learning rate curve**:
```
LR
 ↑
1.0e-4 ┤         ╭─────╮
       │        ╱       ╲
       │       ╱         ╲
       │      ╱           ╲
       │     ╱             ╲
       │    ╱               ╲___
1.0e-6 ┤───╱                    ╲___
       └─────────────────────────────→ Epoch
       0   10    50   100  150  200  250

       ←Warmup→←── Cosine Decay ────→
```

**Tại sao warmup?**
```
Without warmup:
Epoch 0: Large gradients + high LR → unstable
→ Model diverges hoặc converges poorly

With warmup:
Epoch 0-10: Gradual LR increase
→ Stable initialization
→ Better convergence
```

### Alternative: ReduceLROnPlateau

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',           # Maximize Dice score
    factor=0.5,           # LR = LR × 0.5
    patience=10,          # Wait 10 epochs
    verbose=True,
    min_lr=1e-6
)

# Usage trong training loop
val_dice = val_metrics['val_dice_mean']
scheduler.step(val_dice)
```

---

## 5. Mixed Precision Training

### Automatic Mixed Precision (AMP)

**Tại sao AMP?**
```
Float32 (FP32):
- High precision
- 4 bytes per value
- Slow computation

Float16 (FP16):
- Lower precision
- 2 bytes per value
- 2× faster on modern GPUs
- Problem: Numerical instability (underflow/overflow)

AMP Solution:
- Most operations in FP16 (fast)
- Critical operations in FP32 (stable)
- Loss scaling để prevent underflow
```

### Implementation

```python
# Enable AMP
use_amp = True
amp_dtype = 'float16'  # hoặc 'bfloat16' cho A100

# Scaler cho FP16 (không cần cho BF16)
if amp_dtype == 'float16':
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = None

# Training loop
for images, masks, labels in train_loader:
    # Forward trong mixed precision context
    with torch.cuda.amp.autocast(
        enabled=use_amp,
        dtype=torch.float16 if amp_dtype == 'float16' else torch.bfloat16
    ):
        seg_logits, cls_logits, aux = model(images)
        loss = compute_loss(seg_logits, cls_logits, masks, labels)
    
    # Backward với scaling (FP16 only)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    
    optimizer.zero_grad()
```

### Loss Scaling

**Tại sao cần loss scaling?**
```
FP16 range: ~6e-8 to ~6e4

Gradients thường rất nhỏ:
grad = 1e-7  → Underflow in FP16! → grad = 0

Solution: Scale loss lên trước backward
scaled_loss = loss × 2^16  (scale=65536)
scaled_grad = grad × 2^16 = 1e-7 × 65536 = 0.0066
→ No underflow!

Sau đó unscale gradients trước optimizer step
```

### BFloat16 vs Float16

```python
# Float16 (FP16)
# [sign: 1 bit][exponent: 5 bits][mantissa: 10 bits]
# Range: ~6e-8 to 6e4
# Precision: ~3 decimal digits
# Needs: Loss scaling
# Support: V100, RTX 20xx+

# BFloat16 (BF16)
# [sign: 1 bit][exponent: 8 bits][mantissa: 7 bits]
# Range: Same as FP32 (~1e-38 to 3e38)
# Precision: ~2 decimal digits
# Needs: No scaling!
# Support: A100, H100

# Recommendation:
# A100/H100: Use BF16 (simpler, no scaling)
# V100/RTX: Use FP16 với scaling
```

---

## 6. Gradient Accumulation

### Motivation

**Problem**: Large batch size không fit vào GPU
```
Desired batch size: 64
GPU memory: 16GB
Actual fit: 12 samples

→ Cannot use batch size 64!
```

**Solution**: Gradient accumulation
```
Effective batch size = batch_size × accumulation_steps

batch_size = 12
accumulation_steps = 6
→ Effective batch = 12 × 6 = 72
```

### Implementation

```python
gradient_accumulation_steps = 4

for batch_idx, (images, masks, labels) in enumerate(train_loader):
    # Forward
    with torch.cuda.amp.autocast(enabled=use_amp):
        outputs = model(images)
        loss = compute_loss(outputs, masks, labels)
        
        # Normalize loss by accumulation steps
        loss = loss / gradient_accumulation_steps
    
    # Backward (accumulate gradients)
    if use_amp:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    
    # Optimizer step mỗi N batches
    if (batch_idx + 1) % gradient_accumulation_steps == 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad()
```

**Tại sao normalize loss?**
```
Không normalize:
Batch 1: loss=0.5 → backward → grad=g1
Batch 2: loss=0.5 → backward → grad=g2
Batch 3: loss=0.5 → backward → grad=g3
Batch 4: loss=0.5 → backward → grad=g4
→ Total grad = g1+g2+g3+g4 (4× lớn hơn normal!)

Normalize (loss/4):
Batch 1: loss=0.125 → grad=g1/4
Batch 2: loss=0.125 → grad=g2/4
...
→ Total grad = (g1+g2+g3+g4)/4 (correct!)
```

---

## 7. Checkpointing

### Save Checkpoint

```python
def save_checkpoint(
    model, optimizer, scheduler, epoch, metrics, 
    checkpoint_path, is_best=False
):
    """
    Save training checkpoint
    
    Args:
        model: BrainTumNet
        optimizer: AdamW
        scheduler: LR scheduler
        epoch: Current epoch
        metrics: Dict of metrics
        checkpoint_path: Save path
        is_best: Whether this is best model
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'best_val_dice': metrics.get('val_dice_mean', 0.0)
    }
    
    # Save
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as best if applicable
    if is_best:
        best_path = checkpoint_path.replace('last', 'best')
        torch.save(checkpoint, best_path)
        print(f"✓ Saved best model: {best_path}")
```

### Load Checkpoint

```python
def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None):
    """
    Load checkpoint
    
    Returns:
        start_epoch: Epoch để resume training
        best_val_dice: Best validation Dice
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer (optional)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler (optional)
    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_dice = checkpoint.get('best_val_dice', 0.0)
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Best Val Dice: {best_val_dice:.4f}")
    
    return start_epoch, best_val_dice
```

---

## 8. Monitoring và Logging

### TensorBoard Logging

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_name')

# Scalars
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Dice/WT', wt_dice, epoch)
writer.add_scalar('Dice/TC', tc_dice, epoch)
writer.add_scalar('Dice/ED', ed_dice, epoch)
writer.add_scalar('LearningRate', current_lr, epoch)

# Images (visualize predictions)
writer.add_image('Image/FLAIR', flair_img, epoch)
writer.add_image('Mask/Ground Truth', gt_mask, epoch)
writer.add_image('Mask/Prediction', pred_mask, epoch)

# Histograms
for name, param in model.named_parameters():
    writer.add_histogram(f'Parameters/{name}', param, epoch)
    if param.grad is not None:
        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
```

### CSV Logging

```python
import csv

# Initialize CSV
csv_file = 'logs/metrics_fold0.csv'
headers = ['epoch', 'train_loss', 'val_loss', 'wt_dice', 'tc_dice', 'ed_dice', 'mean_dice']

with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()

# Log mỗi epoch
def log_to_csv(epoch, metrics):
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow({
            'epoch': epoch,
            'train_loss': metrics['train_loss'],
            'val_loss': metrics['val_loss'],
            'wt_dice': metrics['val_dice_wt'],
            'tc_dice': metrics['val_dice_tc'],
            'ed_dice': metrics['val_dice_ed'],
            'mean_dice': metrics['val_dice_mean']
        })
```

### Console Logging

```python
import logging

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Log messages
logger.info(f"Starting training...")
logger.info(f"Epoch {epoch}/{num_epochs}")
logger.info(f"Train Loss: {train_loss:.4f}")
logger.info(f"Val Dice: WT={wt:.4f}, TC={tc:.4f}, ED={ed:.4f}")
```

---

**[← Phần 4: Loss Functions](v_04_LOSS_FUNCTIONS.md)** | **[Phần 6: Configuration →](v_06_CONFIGURATION.md)**
