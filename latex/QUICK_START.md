# Quick Start Guide - AMT-UNet Paper

## What's Been Created

A complete research paper for AMT-UNet (Adaptive Masked Transformer U-Net) following Springer LNCS/LNAI format (12-15 pages) with:

### ✅ Complete Files

1. **main.tex** - Main document with LNCS formatting
2. **sections/introduction.tex** - Introduction and motivation
3. **sections/related_work.tex** - Literature review
4. **sections/methodology.tex** - **VERY DETAILED** technical methodology
5. **sections/experiments.tex** - Experiments with data description
6. **sections/conclusion.tex** - Conclusion and future work
7. **references.bib** - 25+ bibliography entries
8. **README.md** - Full compilation and submission guide
9. **TECHNICAL_SUMMARY.md** - Quick reference for all formulas and specs

### 📋 Paper Structure

- **Abstract**: Multi-task hybrid CNN-Transformer for brain tumor analysis
- **Section 1 - Introduction**: Problem, challenges, contributions
- **Section 2 - Related Work**: U-Net variants, transformers, multi-task learning, loss functions
- **Section 3 - Methodology** ⭐ **MOST DETAILED SECTION**:
  - AMT-UNet architecture (encoder, transformer, decoder)
  - Adaptive masked transformer with soft masking
  - ROI-guided classification network
  - Combined loss function (Dice + Focal + IoU + Boundary)
  - Deep supervision strategy
  - Training configuration (optimizer, LR schedule, augmentation)
- **Section 4 - Experiments**:
  - BraTS 2020 dataset description
  - Preprocessing pipeline
  - Implementation details
  - Detailed figure descriptions (6 figures)
  - **[TODO]** Results tables (need to fill after training)
  - **[TODO]** Figures (need to create)
- **Section 5 - Conclusion**: Summary, contributions, future directions

## ⚠️ What Needs To Be Done

### 1. Download LNCS Class File

```bash
# Download from Springer
wget https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines
# Extract llncs.cls and place in latex/ folder
```

### 2. Fill in Results (After Training Completes)

Search for `[TODO]` in [experiments.tex](sections/experiments.tex):

- **Table 1**: Comparison with SOTA methods (U-Net, nnU-Net, TransUNet, Swin-Unet)
- **Table 2**: Classification performance
- **Table 3**: Computational efficiency

### 3. Create Figures

Need to create 6 figures:

1. **Figure 1: Architecture diagram**
   - Show encoder → adaptive masked transformer → decoder → multi-task heads
   - Highlight skip connections with CBAM and multi-scale fusion

2. **Figure 2: Qualitative segmentation results**
   - Multiple example cases (easy, medium, hard)
   - Layout: Input modalities | Ground truth | Prediction
   - Use Python/matplotlib to generate

3. **Figure 3: Attention visualization**
   - CBAM attention at different decoder levels
   - Transformer soft masks highlighting tumor regions
   - ROI gating masks for classification

4. **Figure 4: Feature space visualization**
   - t-SNE of transformer bottleneck features
   - Color code by: tumor grade, size, region type

5. **Figure 5: Training curves**
   - Loss components (Dice, Focal, IoU, Boundary) over epochs
   - Validation Dice scores for WT, TC, ED
   - Learning rate schedule

6. **Figure 6: Per-case analysis**
   - Dice score distributions (box plots)
   - Stratified by tumor size, grade, location

Save figures in `latex/figures/` as PDF (vector) or high-res PNG.

### 4. Update Author Info

In [main.tex](main.tex):

```latex
\author{Your Name\inst{1}\orcidID{0000-0000-0000-0000}}
\institute{Your Institution\\
\email{your.email@institution.edu}}
```

### 5. Add Acknowledgments

In [conclusion.tex](sections/conclusion.tex), section "Acknowledgments":
- Funding sources
- Computing resources
- BraTS dataset citation
- Colleagues

## 🔨 Compilation

### Option 1: Command Line

```bash
cd latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Option 2: Overleaf (Easiest)

1. Upload all files to Overleaf
2. Set compiler to pdfLaTeX
3. Click "Recompile"

### Option 3: LaTeXmk (Automated)

```bash
cd latex
latexmk -pdf main.tex
```

## 📊 Key Highlights of Methodology Section

The methodology section is **extremely detailed** with exact specifications from code and clear design motivations:

### Architecture Details
- Exact channel dimensions at each layer
- Residual blocks with instance normalization
- Motivation for each design choice (e.g., why instance norm, why strided conv)
- Learned downsampling via strided convolutions
- CBAM attention mathematics
- Multi-scale fusion formulation

### Adaptive Masked Transformer Bottleneck
- Patch embedding process
- Soft masking mechanism with complete formulas
- Content-based attention weighting
- Multi-head self-attention with masking
- Feed-forward network structure
- All dimensions and hyperparameters

### Loss Functions with Motivations
- Complete mathematical formulations for all 4 loss components
- Exact weights with justifications: Dice=1.0, Focal=1.0, IoU=2.5, Boundary=0.6
- Focal parameters: γ=3.0, α=[0.0, 0.4, 0.3]
- Class weights: [1.0, 3.0, 4.0]
- Deep supervision weight: 0.3
- Multi-task weights: seg=1.0, cls=0.5
- WHY each loss component and weight was chosen

### Training Strategy
- AdamW with exact hyperparameters
- Cosine annealing with warm-up formula
- Mixed precision training
- Gradient clipping
- Data augmentation specifications

All formulas and motivations match the actual code implementation exactly.

## 📁 File Organization

```
latex/
├── main.tex                    # Main document
├── sections/
│   ├── introduction.tex        # Section 1
│   ├── related_work.tex        # Section 2
│   ├── methodology.tex         # Section 3 ⭐ VERY DETAILED
│   ├── experiments.tex         # Section 4 (has TODOs)
│   └── conclusion.tex          # Section 5
├── figures/                    # [TODO] Create this folder and add figures
├── references.bib              # Bibliography (25+ entries)
├── README.md                   # Full guide
├── TECHNICAL_SUMMARY.md        # Quick reference
└── QUICK_START.md             # This file
```

## ✅ Checklist Before Submission

- [ ] Download and place `llncs.cls` in latex folder
- [ ] Fill all `[TODO]` placeholders with actual results
- [ ] Create all 6 figures
- [ ] Update author name and affiliation
- [ ] Add acknowledgments
- [ ] Compile successfully to PDF
- [ ] Check page count (should be 12-15 pages)
- [ ] Proofread entire paper
- [ ] Verify all references are correct
- [ ] Check all cross-references work

## 📖 Paper Specifications

- **Format**: Springer LNCS/LNAI
- **Length**: 12-15 pages (currently structured to fit)
- **Language**: English (US spelling)
- **References**: 25+ citations (comprehensive)
- **Figures**: 6 required (placeholders marked)
- **Tables**: 4 required (placeholders with TODOs)

## 💡 Tips

1. **Methodology is complete with motivations** - All technical details with WHY explanations
2. **Single variant focus** - 37M parameter model (no ablation study needed)
3. **Only results need filling** - Tables have clear TODO markers
4. **Figures are well-specified** - 6 detailed figure descriptions provided
5. **Bibliography is comprehensive** - All major related works included
6. **Code-based accuracy** - Every formula matches implementation exactly

## 📞 Support

If you need help:
1. Check [README.md](README.md) for detailed compilation instructions
2. Check [TECHNICAL_SUMMARY.md](TECHNICAL_SUMMARY.md) for quick formula reference
3. All formulas extracted from actual code implementation

## 🎯 Next Steps

1. **Download llncs.cls** from Springer website
2. **Test compile** to ensure LaTeX works correctly
3. **Complete model training** to obtain results
4. **Fill TODO placeholders** in experiments.tex with actual numbers
5. **Create 6 figures** as specified in detailed descriptions
6. **Update author information** and acknowledgments
7. **Final compile and proofread** before submission

The paper structure is **complete** - AMT-UNet is presented as a unified method with clear motivations. Only experimental results and figures need to be added after training!
