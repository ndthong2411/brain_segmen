# Brain Tumor Segmentation Project

This repository contains the **AMT-UNet** project - a hybrid CNN-transformer architecture for brain tumor segmentation and classification.

## Project Location

**All project files are located in the `braintumnet/` directory.**

```
brain_segmen/
├── braintumnet/          ← ALL PROJECT FILES HERE
│   ├── README.md         ← Main documentation
│   ├── configs/          ← Configuration files
│   ├── scripts/          ← Training, evaluation, inference
│   ├── src/              ← Core package
│   ├── data/             ← Datasets
│   ├── docs/             ← Documentation
│   └── ...
├── old/                  ← Legacy files (archived)
└── brats2020_data/       ← Original dataset
```

## Getting Started

### Navigate to Project Directory
```bash
cd braintumnet
```

### Read the Documentation
```bash
# View the main README
cat README.md

# Or open in your editor/browser
```

### Quick Setup Verification
```bash
cd braintumnet
python verify_setup.py
```

### Install Dependencies
```bash
cd braintumnet
pip install -r requirements.txt
```

### Train a Model
```bash
cd braintumnet
python scripts/train.py --cfg configs/quick_test.yaml --fold 0
```

## Documentation

- **Main README**: `braintumnet/README.md` - Complete project documentation
- **Test Results**: `braintumnet/docs/TEST_RESULTS.md` - Detailed test report
- **Verification**: `braintumnet/docs/VERIFICATION_SUMMARY.md` - System verification
- **Contributing**: `braintumnet/AGENTS.md` - Contributor guidelines

## Key Features

✅ Multi-task learning (segmentation + classification)
✅ Advanced architecture (U-Net + CBAM + Transformer)
✅ HDF5 BraTS2020 support
✅ Complete training pipeline
✅ TensorBoard logging
✅ Production ready

## Status

**✅ ALL SYSTEMS OPERATIONAL**

All components tested and verified. Ready for production use.

---

**For complete documentation, see: `braintumnet/README.md`**
