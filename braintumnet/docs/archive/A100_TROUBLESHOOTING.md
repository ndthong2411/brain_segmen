# A100 Training Troubleshooting Guide

## Quick Fixes Applied

### ✅ Issue 1: `torch.compile()` Failed with `ptxas` Error

**Error:**
```
RuntimeError: `ptxas` failed with error code -2
```

**Root Cause:**
- Triton compiler (used by `torch.compile()`) has issues with some CUDA/system configurations
- `ptxas` is NVIDIA's PTX assembler, compilation failed

**Fix Applied:**
- Disabled `torch.compile()` in [full_dataset_multimodal_a100.yaml:30](../configs/full_dataset_multimodal_a100.yaml#L30)
- Changed: `use_compile: true` → `use_compile: false`

**Impact:**
- Lost 10-20% speedup from compilation
- Still get 3-4x speedup from other optimizations
- Training will work reliably

**Alternative Solutions (if you want to try enabling later):**
```bash
# Option 1: Update PyTorch/Triton
pip install --upgrade torch torchvision triton

# Option 2: Use different compile mode
use_compile: true
compile_mode: "reduce-overhead"  # Instead of "max-autotune"

# Option 3: Set environment variable to disable Triton
export TORCH_COMPILE_DISABLE=1
```

---

### ✅ Issue 2: tqdm Progress Bars Not Showing

**Symptoms:**
- No visible progress bars during training
- Can't see iteration progress

**Fixes Applied:**

1. **Explicit stdout routing** ([trainer.py:179-180](../src/braintumnet/engine/trainer.py#L179))
   ```python
   pbar = tqdm(..., file=sys.stdout, leave=True, ncols=120)
   ```

2. **Added fallback printing** ([trainer.py:228-232](../src/braintumnet/engine/trainer.py#L228))
   - If tqdm doesn't work, prints every 10 batches
   - Ensures you always see progress

**Why tqdm might not work:**
- Redirected stdout (logging systems)
- Jupyter notebooks vs terminal
- SSH sessions without proper TTY
- Some job schedulers (SLURM, PBS)

**Manual check if tqdm is working:**
```python
from tqdm import tqdm
import time
for i in tqdm(range(100)):
    time.sleep(0.01)
# Should see progress bar
```

**Environment-specific fixes:**

For SLURM:
```bash
# In your SLURM script
export TQDM_DISABLE=0
export PYTHONUNBUFFERED=1
srun python scripts/train.py --cfg configs/full_dataset_multimodal_a100.yaml
```

For SSH/screen/tmux:
```bash
# Ensure proper terminal
export TERM=xterm-256color
python scripts/train.py --cfg configs/full_dataset_multimodal_a100.yaml
```

---

## Common A100 Training Issues

### Issue: Out of Memory (OOM)

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions (in order of preference):**

1. **Reduce batch size:**
   ```yaml
   batch_size: 64  # from 96
   val_batch_size: 96  # from 128
   ```

2. **Enable gradient accumulation:**
   ```yaml
   batch_size: 48
   grad_accum_steps: 2  # Effective batch = 48 * 2 = 96
   ```

3. **Reduce model size:**
   ```yaml
   base: 24  # from 32
   dim: 192  # from 256
   ```

4. **Disable channels_last (small impact):**
   ```yaml
   use_channels_last: false
   ```

**Debug memory usage:**
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

---

### Issue: Low GPU Utilization (<70%)

**Check with:**
```bash
nvidia-smi dmon -s mu
# Watch GPU utilization and memory
```

**Solutions:**

1. **Increase workers:**
   ```yaml
   workers: 12  # from 10
   ```

2. **Increase batch size:**
   ```yaml
   batch_size: 128  # if memory allows
   ```

3. **Check data loading bottleneck:**
   ```python
   # Add to trainer.py temporarily
   import time
   data_time = 0
   compute_time = 0

   for batch in train_loader:
       t0 = time.time()
       img = batch["image"].to(device)
       t1 = time.time()
       seg, cls = model(img)
       loss.backward()
       t2 = time.time()

       data_time += (t1 - t0)
       compute_time += (t2 - t1)

   print(f"Data loading: {data_time:.2f}s, Compute: {compute_time:.2f}s")
   # If data_time > compute_time, increase workers
   ```

---

### Issue: NaN Loss

**Symptoms:**
```
loss: nan
```

**Solutions:**

1. **Reduce learning rate:**
   ```yaml
   lr: 1.5e-4  # from 2.25e-4
   ```

2. **Increase gradient clipping:**
   ```yaml
   grad_clip_norm: 0.5  # from 1.0
   ```

3. **Check for mixed precision issues:**
   ```yaml
   amp: false  # Disable temporarily to debug
   ```

4. **Add gradient checking:**
   ```python
   # In trainer.py, after backward()
   for name, param in model.named_parameters():
       if param.grad is not None:
           if torch.isnan(param.grad).any():
               print(f"NaN gradient in {name}")
   ```

---

### Issue: Training Too Slow (not hitting 3-5x speedup)

**Expected speeds:**
- **Before optimizations:** ~100 it/s
- **After optimizations:** ~400-500 it/s

**Checklist:**

1. ✅ **Verify optimizations are active:**
   ```python
   import torch
   print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")  # True
   print(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")  # True
   ```

2. ✅ **Check GPU utilization:**
   ```bash
   nvidia-smi dmon -s mu
   # Should be >90%
   ```

3. ✅ **Check config:**
   ```bash
   cat configs/full_dataset_multimodal_a100.yaml | grep -E "batch_size|workers|use_"
   # batch_size: 96
   # workers: 10
   # use_compile: false (OK for stability)
   # use_channels_last: true
   ```

4. ✅ **Profile data loading:**
   ```python
   # Temporarily in trainer.py
   import time
   start = time.time()
   for i, batch in enumerate(train_loader):
       if i == 10:
           break
   print(f"10 batches in {time.time()-start:.2f}s")
   # Should be <2 seconds
   ```

---

### Issue: Validation Taking Too Long

**Solution 1: Skip validation in early epochs**
```yaml
val_interval: 2  # Validate every 2 epochs
```

**Solution 2: Increase validation batch size**
```yaml
val_batch_size: 192  # from 128 (no gradients = more memory)
```

**Solution 3: Reduce validation frequency during training**
```python
# In trainer.py, modify validation logic
if epoch < 20:
    val_interval = 3  # Every 3 epochs early on
elif epoch < 50:
    val_interval = 2  # Every 2 epochs mid-training
else:
    val_interval = 1  # Every epoch late training
```

---

### Issue: Disk I/O Bottleneck

**Symptoms:**
- High `iowait` in `top`
- GPU underutilized despite enough workers

**Solutions:**

1. **Copy data to local SSD:**
   ```bash
   # If on compute cluster with local scratch
   cp -r data/processed_full_multimodal /tmp/
   # Update config
   proc_root: "/tmp/processed_full_multimodal"
   ```

2. **Use RAM disk (if enough RAM):**
   ```bash
   sudo mkdir /mnt/ramdisk
   sudo mount -t tmpfs -o size=50G tmpfs /mnt/ramdisk
   cp -r data/processed_full_multimodal /mnt/ramdisk/
   ```

3. **Increase prefetch_factor:**
   ```python
   # In trainer.py
   prefetch_factor=4  # from 2
   ```

---

## Environment Setup Issues

### Issue: CUDA Out of Date

**Check CUDA version:**
```bash
nvcc --version
nvidia-smi
```

**A100 requires:**
- CUDA >= 11.1 (for A100 support)
- CUDA >= 11.8 (for TF32 full support)
- CUDA >= 12.0 (recommended)

**If CUDA is old:**
```bash
# Install newer CUDA toolkit
# OR use conda
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

---

### Issue: Driver Issues

**Symptoms:**
```
CUDA driver version is insufficient for CUDA runtime version
```

**Check:**
```bash
nvidia-smi  # Driver version
cat /usr/local/cuda/version.txt  # CUDA version
```

**Fix:**
```bash
# Update NVIDIA driver
sudo apt update
sudo apt install nvidia-driver-535  # Or latest
sudo reboot
```

---

## Monitoring & Debugging Commands

### Real-time GPU monitoring
```bash
# Option 1: nvidia-smi
watch -n 0.5 nvidia-smi

# Option 2: nvtop (better visualization)
nvtop

# Option 3: nvidia-smi dmon (detailed)
nvidia-smi dmon -s mu -d 1
```

### Check training logs
```bash
# Live training log
tail -f logs/braintumnet_full_multimodal_fold0.log

# Check for errors
grep -i "error\|warning\|nan" logs/*.log

# Check metrics
tail -20 logs/braintumnet_full_multimodal_fold0_metrics.csv
```

### TensorBoard monitoring
```bash
# Start TensorBoard
tensorboard --logdir=runs --port=6006

# SSH tunnel (if remote)
ssh -L 6006:localhost:6006 user@server

# Open browser
http://localhost:6006
```

---

## Performance Benchmarking

### Quick benchmark script
```python
import torch
import time
from braintumnet.models.braintumnet import BrainTumNet

device = "cuda"
model = BrainTumNet(in_ch=4, num_cls=2).to(device)
model = model.to(memory_format=torch.channels_last)

# Warmup
for _ in range(10):
    x = torch.randn(96, 4, 256, 256, device=device).to(memory_format=torch.channels_last)
    seg, cls = model(x)

# Benchmark
times = []
for _ in range(100):
    x = torch.randn(96, 4, 256, 256, device=device).to(memory_format=torch.channels_last)
    torch.cuda.synchronize()
    t0 = time.time()
    seg, cls = model(x)
    torch.cuda.synchronize()
    times.append(time.time() - t0)

print(f"Average forward time: {sum(times)/len(times)*1000:.2f} ms")
print(f"Throughput: {96 / (sum(times)/len(times)):.1f} samples/sec")
```

**Expected results on A100:**
- Forward time: ~30-50ms per batch (96 samples)
- Throughput: ~2000-3000 samples/sec

---

## Quick Reference

### Key Config Settings
```yaml
# Optimal for A100-80GB
batch_size: 96
val_batch_size: 128
workers: 10
lr: 2.25e-4
use_compile: false  # Due to ptxas issue
use_channels_last: true
grad_clip_norm: 1.0
log_interval: 50
```

### Key Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=0
```

### Emergency Fallback Config
```yaml
# If all else fails, use this minimal config
batch_size: 32
workers: 4
lr: 1.0e-4
use_compile: false
use_channels_last: false
amp: false
```

---

## Getting Help

1. **Check logs first:**
   ```bash
   tail -50 logs/braintumnet_full_multimodal_fold0.log
   ```

2. **Verify environment:**
   ```python
   import torch
   print(f"PyTorch: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   print(f"GPU: {torch.cuda.get_device_name(0)}")
   ```

3. **Test minimal training:**
   ```bash
   # Use quick test config first
   python scripts/train.py --cfg configs/multimodal_quick_test.yaml --fold 0
   # If this works, problem is with full dataset config
   ```

---

**Last Updated:** 2025-10-08
**Related:** [A100_OPTIMIZATION_GUIDE.md](A100_OPTIMIZATION_GUIDE.md)
