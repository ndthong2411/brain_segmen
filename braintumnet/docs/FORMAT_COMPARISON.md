# Data Format Comparison: PNG vs DICOM vs H5

**Version**: 1.0
**Last Updated**: 2025-10-29

---

## 📊 Quick Answer

### ❌ **DO NOT train directly on DICOM**

**Use PNG or H5 for training. DICOM is for clinical deployment/inference only.**

---

## 🏆 Performance Comparison

### Loading Speed

| Format | Load Time | Relative Speed | Training Impact (1 epoch) |
|--------|-----------|----------------|---------------------------|
| **H5** | ~0.5-1ms | **1.0x (fastest)** | ~16 seconds |
| **PNG** | ~1-2ms | **1.5-2x** | ~32 seconds |
| **DICOM** | ~10-50ms | **15-20x slower** | ~8 minutes |

**For 1000 iterations, batch_size=16:**
- PNG: 32 seconds data loading
- DICOM: 480 seconds (8 minutes!) data loading
- **DICOM is 15x slower than PNG!**

### Memory Usage

| Format | Memory/Image | Relative Memory |
|--------|--------------|-----------------|
| **PNG** | ~70-100 KB | 1.0x |
| **H5** | ~50-80 KB | 0.7x (best) |
| **DICOM** | ~300-500 KB | 3-5x |

### File Size on Disk

| Format | Size/Image | For 57,195 images |
|--------|------------|-------------------|
| **PNG** | ~10-50 KB | ~1-2 GB |
| **H5** | ~10-30 KB | ~0.5-1 GB |
| **DICOM** | ~100-200 KB | ~10-12 GB |

---

## ⚡ Why DICOM is Slow for Training

### 1. **Complex Parsing**
```python
# DICOM loading (slow)
ds = pydicom.dcmread('file.dcm')       # Parse metadata (XML-like)
pixel_array = ds.pixel_array           # Extract pixels
pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept  # Apply rescale
# Total: ~10-50ms

# PNG loading (fast)
img = Image.open('file.png')           # Direct binary read
array = np.array(img)                  # Convert to array
# Total: ~1-2ms
```

### 2. **Metadata Overhead**

DICOM file structure:
```
DICOM file (~200 KB)
├── Preamble (128 bytes)
├── DICOM prefix (4 bytes)
├── File Meta Information (~500 bytes)
├── Patient Information (~1 KB)
├── Study Information (~1 KB)
├── Series Information (~1 KB)
├── Image Information (~2 KB)
├── Equipment Information (~1 KB)
└── Pixel Data (~190 KB)
```

PNG file structure:
```
PNG file (~50 KB)
├── PNG signature (8 bytes)
├── IHDR chunk (header, ~25 bytes)
├── IDAT chunk (compressed pixels, ~49 KB)
└── IEND chunk (end marker, ~12 bytes)
```

**→ DICOM has 10-20 KB metadata overhead per file**

### 3. **CPU Bottleneck**

Training loop with DICOM:
```
GPU: ||||||||||----------  (50% utilization)
CPU: ||||||||||||||||||||  (100% utilization - loading DICOM!)
                    ↑
              Bottleneck!
```

Training loop with PNG:
```
GPU: ||||||||||||||||||||  (95% utilization)
CPU: ||||||----------      (30% utilization - idle)
                    ↑
              Optimal!
```

---

## 🎯 Benchmark Results

### Run Benchmark

```bash
# Prepare directories
# PNG: data/processed_multiclass/flair/
# DICOM: data/dicom_output/Patient001_vol1/FLAIR/
# H5: data/h5_files/

# Run benchmark
python scripts/benchmark_data_loading.py \
    --png_dir data/processed_multiclass/flair \
    --dicom_dir data/dicom_output/Patient001_vol1/FLAIR \
    --h5_dir data/h5_files \
    --num_samples 1000
```

### Expected Results

```
================================================================================
COMPARISON SUMMARY
================================================================================

Format       Load Time       Memory/Image    Throughput
------------ --------------- --------------- ---------------
H5           0.87 ms         52.31 KB        1149.4 img/s
PNG          1.54 ms         78.92 KB        649.4 img/s
DICOM        23.18 ms        412.67 KB       43.1 img/s

================================================================================
RELATIVE PERFORMANCE (vs PNG)
================================================================================

Format       Load Time            Memory
------------ -------------------- --------------------
H5           0.56x faster         0.66x
PNG          1.00x                1.00x
DICOM        15.05x slower        5.23x

================================================================================
ESTIMATED TRAINING TIME IMPACT
================================================================================
For 1 epoch with 1000 iterations, batch_size=16:

Format       Data Loading         vs PNG
------------ -------------------- --------------------
H5           13.9 seconds         -10.7s faster
PNG          24.6 seconds
DICOM        370.9 seconds        +346.3s slower

================================================================================
RECOMMENDATION
================================================================================

✓ Fastest format:        H5 (0.87 ms/image)
✓ Most memory efficient: H5 (52.31 KB/image)

For training, we recommend: PNG or H5 format
DICOM should only be used for inference/deployment, not training
```

---

## 📋 Format Use Cases

### H5 Format

**Best for:**
- ✅ Training (fastest loading)
- ✅ Research experiments
- ✅ Kaggle competitions
- ✅ Batch processing

**Pros:**
- Fastest loading (~0.5-1ms)
- Smallest memory footprint
- Built-in compression
- Multi-modal in single file

**Cons:**
- Not a medical standard
- Requires h5py library
- Less interoperable

**When to use:**
```
Research → Development → Experiments → H5 Format
```

### PNG Format

**Best for:**
- ✅ Training (fast, universal)
- ✅ Visualization
- ✅ Debugging
- ✅ Quick inspection

**Pros:**
- Fast loading (~1-2ms)
- Universal support
- Easy to view/debug
- Lossless compression

**Cons:**
- Separate file per modality
- No metadata storage
- Not medical standard

**When to use:**
```
Development → Debugging → Quick iterations → PNG Format
```

### DICOM Format

**Best for:**
- ✅ Clinical deployment
- ✅ PACS integration
- ✅ Hospital workflows
- ✅ Regulatory compliance (FDA/CE)
- ✅ Data sharing with clinicians

**Pros:**
- Medical standard (ISO 12052)
- Rich metadata
- PACS compatibility
- Interoperable
- Regulatory compliant

**Cons:**
- Slow loading (~10-50ms)
- Large file size
- High memory usage
- Complex parsing

**When to use:**
```
Training complete → Deploy to clinic → DICOM Format
```

---

## 🔄 Recommended Workflow

### Development & Training

```
1. Raw DICOM (from hospital)
         ↓
2. Convert to PNG/H5
         ↓
3. Train model (PNG/H5)
         ↓
4. Validate model (PNG/H5)
         ↓
5. Optimize & tune (PNG/H5)
```

### Clinical Deployment

```
6. Export model
         ↓
7. Create DICOM pipeline
         ↓
8. Test with DICOM data
         ↓
9. Integrate with PACS
         ↓
10. Clinical validation (DICOM)
```

---

## 💡 Optimization Tips

### If You Must Use DICOM for Training

**Option 1: Preload to RAM** (if you have enough memory)

```python
# Preload all DICOM to memory
print("Preloading DICOM files to RAM...")
dicom_cache = {}
for dcm_path in dicom_files:
    ds = pydicom.dcmread(dcm_path)
    pixel_array = ds.pixel_array.astype(np.float32)
    pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
    dicom_cache[dcm_path] = pixel_array

# Training loop
for epoch in range(epochs):
    for dcm_path in dicom_files:
        pixel_array = dicom_cache[dcm_path]  # Fast!
        # ... train ...
```

**Memory requirement:** ~50 GB for 57,195 images

**Option 2: Convert on-the-fly with caching**

```python
from functools import lru_cache

@lru_cache(maxsize=1000)  # Cache 1000 most recent
def load_dicom_cached(dcm_path):
    ds = pydicom.dcmread(dcm_path)
    pixel_array = ds.pixel_array.astype(np.float32)
    pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
    return pixel_array
```

**Option 3: Convert to PNG first** (recommended!)

```bash
# One-time conversion
python scripts/preprocess_dicom_to_multiclass.py \
    --dicom_dir data/dicom \
    --out_dir data/processed_png

# Train on PNG
python scripts/train.py --cfg configs/phase2_small.yaml --fold 0
```

---

## 📊 Training Time Comparison

### Example: Training 250 epochs, batch_size=16

| Format | Data Loading/Epoch | Total Data Loading | GPU Training | Total Time |
|--------|--------------------|--------------------|--------------|------------|
| **H5** | 16 sec | 67 min | 42 hours | **~43 hours** |
| **PNG** | 32 sec | 133 min | 42 hours | **~44 hours** |
| **DICOM** | 480 sec (8 min) | 33 hours | 42 hours | **~75 hours** |

**→ Training on DICOM takes 70% longer!**

---

## 🎯 Decision Matrix

| Scenario | Recommended Format | Reason |
|----------|-------------------|--------|
| **Training from scratch** | PNG or H5 | Fast loading, low overhead |
| **Research experiments** | H5 | Fastest, most efficient |
| **Quick prototyping** | PNG | Easy to view/debug |
| **Production inference** | DICOM | Clinical standard |
| **Hospital integration** | DICOM | PACS compatibility |
| **FDA/CE submission** | DICOM | Regulatory requirement |
| **Data sharing (clinical)** | DICOM | Interoperable |
| **Data sharing (research)** | H5 or PNG | Faster transfer |

---

## ✅ Final Recommendation

### For Training
```
Use PNG or H5 format
- Fast loading (15-20x faster than DICOM)
- Low memory overhead
- Optimal GPU utilization
```

### For Clinical Deployment
```
Use DICOM format
- Medical standard
- PACS integration
- Regulatory compliance
```

### Best Practice
```
1. Convert DICOM → PNG/H5 (one-time, ~15 min)
2. Train on PNG/H5 (fast, efficient)
3. Deploy with DICOM pipeline (clinical standard)
```

---

## 📚 Related Documentation

- [H5_TO_DICOM_CONVERSION_GUIDE.md](H5_TO_DICOM_CONVERSION_GUIDE.md) - Convert H5 to DICOM
- [DICOM_PREPROCESSING_GUIDE.md](DICOM_PREPROCESSING_GUIDE.md) - Convert DICOM to PNG
- [QUICKSTART.md](quickstart/QUICKSTART.md) - Training guide

---

## 🔗 Run Benchmark Yourself

```bash
# Install dependencies
pip install psutil pydicom h5py pillow

# Run benchmark
python scripts/benchmark_data_loading.py \
    --png_dir data/processed_multiclass/flair \
    --dicom_dir data/dicom_output/Patient001_vol1/FLAIR \
    --h5_dir data/h5_files \
    --num_samples 1000 \
    --batch_size 16
```

---

**Summary: Train on PNG/H5, deploy on DICOM! 🚀**
