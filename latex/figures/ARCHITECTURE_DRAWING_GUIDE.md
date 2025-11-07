# AMT-UNet Architecture Diagram - Drawing Guide

## 📐 Where to Save

Save the final diagram as:
- **Primary:** `latex/figures/architecture.pdf` (vector, best quality)
- **Alternative:** `latex/figures/architecture.png` (300 DPI minimum)

---

## 🎨 Quick Draw Instructions (draw.io / PowerPoint)

### 1. Open draw.io (https://app.diagrams.net/)

### 2. Page Setup:
- Size: A4 Landscape (297mm × 210mm)
- Grid: 10px
- Snap to grid: ON

### 3. Main Components to Draw (Top to Bottom):

---

## 📦 Component Specifications

### INPUT BLOCK
**Position:** Top center
**Size:** 200px × 60px
**Style:** Rounded rectangle, light gray fill (#F0F0F0)
**Text:**
```
INPUT
4-channel MRI
(FLAIR, T1, T1CE, T2)
256 × 256 × 4
```
**Arrow down:** Thick black arrow (10px width)

---

### ENCODER BLOCK (Left side)
**Position:** Below input, left side
**Size:** 180px × 400px
**Style:** Light blue (#ADD8E6) rounded rectangle

**Contains 4 sub-blocks (vertical stack):**

**Block E1:**
```
┌──────────────────┐
│ ResConvBlock     │
│ 48 channels      │
│ 256×256          │
└──────────────────┘
       ↓ (stride 2)
```
**Arrow right:** Dashed line to skip connection

**Block E2:**
```
┌──────────────────┐
│ ResConvBlock     │
│ 96 channels      │
│ 128×128          │
└──────────────────┘
       ↓ (stride 2)
```
**Arrow right:** Dashed line to skip connection

**Block E3:**
```
┌──────────────────┐
│ ResConvBlock     │
│ 192 channels     │
│ 64×64            │
└──────────────────┘
       ↓ (stride 2)
```
**Arrow right:** Dashed line to skip connection

**Block E4:**
```
┌──────────────────┐
│ ResConvBlock     │
│ 384 channels     │
│ 32×32            │
└──────────────────┘
       ↓ (stride 2)
```
**Output:** 384 × 16 × 16

---

### TRANSFORMER BLOCK (Center)
**Position:** Center, below encoder
**Size:** 250px × 200px
**Style:** Light orange (#FFD700) rounded rectangle

**Layout:**
```
┌────────────────────────────────┐
│  ADAPTIVE MASKED TRANSFORMER   │
│  ────────────────────────────  │
│                                │
│  ┌──────────────────────────┐ │
│  │ Patch Embed (8×8 patches)│ │
│  │ 4 tokens × 384 dim       │ │
│  └───────────┬──────────────┘ │
│              ↓                 │
│  ┌──────────────────────────┐ │
│  │ Soft Mask Generator      │ │
│  │ MLP → Sigmoid            │ │
│  │ Output: 8 heads × 4 tok  │ │
│  └───────────┬──────────────┘ │
│              ↓                 │
│  ┌──────────────────────────┐ │
│  │ Masked Self-Attention    │ │
│  │ 4 layers, 8 heads        │ │
│  │ dim = 384                │ │
│  └──────────────────────────┘ │
│                                │
│  Output: 384 × 16 × 16        │
└────────────────────────────────┘
```

---

### DECODER BLOCK (Right side)
**Position:** Right side, mirror of encoder
**Size:** 180px × 400px
**Style:** Light green (#90EE90) rounded rectangle

**Contains 4 sub-blocks (vertical stack, bottom to top):**

**Block D4:**
```
      ┌──────────────────┐
      │ UpConv (stride 2)│
      └────────┬─────────┘
               ↓
    ← CBAM + Skip (from E4)
               ↓
      ┌──────────────────┐
      │ ResConvBlock     │
      │ 384 channels     │
      │ 32×32            │
      └──────────────────┘
```
**Side arrow:** Right arrow → "Aux3 (64×64)"

**Block D3:**
```
      ┌──────────────────┐
      │ UpConv (stride 2)│
      └────────┬─────────┘
               ↓
    ← CBAM + Skip (from E3)
               ↓
      ┌──────────────────┐
      │ ResConvBlock     │
      │ 192 channels     │
      │ 64×64            │
      └──────────────────┘
```
**Side arrow:** Right arrow → "Aux2 (128×128)"

**Block D2:**
```
      ┌──────────────────┐
      │ UpConv (stride 2)│
      └────────┬─────────┘
               ↓
    ← CBAM + Skip (from E2)
               ↓
      ┌──────────────────┐
      │ ResConvBlock     │
      │ 96 channels      │
      │ 128×128          │
      └──────────────────┘
```
**Side arrow:** Right arrow → "Aux1 (256×256)"

**Block D1:**
```
      ┌──────────────────┐
      │ UpConv (stride 2)│
      └────────┬─────────┘
               ↓
    ← CBAM + Skip (from E1)
               ↓
      ┌──────────────────┐
      │ ResConvBlock     │
      │ 48 channels      │
      │ 256×256          │
      └──────────────────┘
```

---

### MULTI-SCALE FUSION BLOCK
**Position:** Below decoder
**Size:** 200px × 80px
**Style:** Light purple (#DDA0DD) rounded rectangle

```
┌─────────────────────────────┐
│   MULTI-SCALE FUSION        │
│  ─────────────────────────  │
│  Interpolate D1, D2, D3, D4 │
│  to 256×256 → Sum → Norm    │
│  Concat with D1             │
└──────────┬──────────────────┘
           ↓
```

---

### OUTPUT BRANCHES (Bottom)

**Split into 2 branches:**

**LEFT BRANCH - Segmentation:**
```
┌──────────────────────┐
│  SEGMENTATION HEAD   │
│  ──────────────────  │
│  ResConvBlock        │
│  Conv 1×1            │
│  → 3 classes         │
│  (BG, TC, ED)        │
└──────────┬───────────┘
           ↓
    Segmentation Output
    (256 × 256 × 3)
```

**RIGHT BRANCH - Classification:**
```
┌──────────────────────┐
│  ROI GATING          │
│  ──────────────────  │
│  Conv 1×1 (4→1)      │
│  × Seg probability   │
│  (gradient detach)   │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  CLASSIFICATION NET  │
│  ──────────────────  │
│  T-Inception         │
│  64 → 128 → 256      │
│  Global Pool         │
│  Dropout (0.3)       │
│  Linear → 2 classes  │
└──────────┬───────────┘
           ↓
   Classification Output
   (HGG, LGG)
```

---

### SKIP CONNECTIONS (Dashed arrows)
**From:** Each encoder block (E1, E2, E3, E4)
**To:** Corresponding decoder block (D1, D2, D3, D4)
**Style:** Dashed gray arrows
**Label on arrow:** "CBAM"

---

### DEEP SUPERVISION (Side arrows)
**From:** Decoder blocks D2, D3, D4
**To:** Right side
**Style:** Thin solid arrows
**Labels:** "Aux1 (256×256)", "Aux2 (128×128)", "Aux3 (64×64)"

---

## 🎨 Color Scheme

| Component | Color | Hex Code |
|-----------|-------|----------|
| Encoder blocks | Light blue | #ADD8E6 |
| Decoder blocks | Light green | #90EE90 |
| Transformer | Light orange | #FFD700 |
| Multi-scale fusion | Light purple | #DDA0DD |
| Segmentation head | Blue | #4169E1 |
| Classification head | Red | #DC143C |
| Skip connections | Gray | #808080 |
| Text | Black | #000000 |
| Borders | Dark gray | #404040 |

---

## 📏 Dimensions Guide

### Component Sizes (in pixels):
- Main encoder/decoder blocks: 180 × 400
- Transformer block: 250 × 200
- Sub-blocks (E1, E2, etc.): 160 × 80
- Output heads: 200 × 120
- Arrows: 10px width for main flow, 5px for skip connections

### Spacing:
- Between encoder and transformer: 50px
- Between transformer and decoder: 50px
- Between blocks vertically: 20px
- Margin around page: 30px

---

## 🔍 Detail Panels (Optional - can be separate figures)

### Panel A: Residual Convolutional Block
```
Input (C_in channels)
       ↓
┌──────────────────────┐
│ Conv 3×3             │
│ InstanceNorm         │
│ LeakyReLU (0.01)     │
│ Dropout (0.15)       │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Conv 3×3             │
│ InstanceNorm         │
└──────────┬───────────┘
           ↓
           + ← Conv 1×1 (if C_in ≠ C_out)
           ↓
      LeakyReLU
           ↓
  Output (C_out channels)
```

### Panel B: CBAM Attention
```
Input (C channels)
       ↓
┌──────────────────────────┐
│  CHANNEL ATTENTION       │
│  AvgPool + MaxPool       │
│  → MLP (C→C/16→C)        │
│  → Sigmoid               │
└───────────┬──────────────┘
            ↓ (× element-wise)
       Intermediate
            ↓
┌──────────────────────────┐
│  SPATIAL ATTENTION       │
│  Channel-wise Max & Avg  │
│  → Conv 7×7              │
│  → Sigmoid               │
└───────────┬──────────────┘
            ↓ (× element-wise)
         Output
```

### Panel C: Masked Transformer Block
```
Input Tokens (N × D)
       ↓
 LayerNorm
       ↓
   Q, K, V = Linear(tokens)
       ↓
    Attention Logits = QK^T/√d
       ↓
       + ← log(mask + ε)  [Soft mask injection]
       ↓
    Softmax
       ↓
    Attention × V
       ↓
    + Residual
       ↓
   LayerNorm
       ↓
    FFN: Linear(D→4D) → GELU → Linear(4D→D)
       ↓
    + Residual
       ↓
  Output Tokens
```

---

## 📝 Text Annotations to Add

### Main diagram annotations:
1. Near input: "Multi-modal MRI: FLAIR (edema), T1 (anatomy), T1CE (active tumor), T2 (fluid)"
2. Near encoder: "Strided convolution downsampling (learned, not max-pooling)"
3. Near transformer: "Adaptive masking focuses attention on tumor-relevant patches"
4. Near skip connections: "CBAM attention refines encoder features"
5. Near decoder: "Transposed convolution upsampling + skip connections"
6. Near multi-scale fusion: "Combines features from all decoder levels"
7. Near ROI gating: "Gradient detach prevents task interference"
8. Near deep supervision: "Auxiliary losses at 3 scales (weight 0.3)"

### Key metrics to show:
- Total parameters: **37M**
- Input size: **256 × 256 × 4**
- Transformer: **4 layers, 8 heads, 384 dim**
- Output: **Segmentation (3 classes) + Classification (2 classes)**

---

## 🖼️ Figure Caption (to add in LaTeX)

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=0.95\textwidth]{figures/architecture.pdf}
\caption{AMT-UNet architecture. The model consists of a CNN encoder-decoder
with residual blocks and instance normalization, an adaptive masked transformer
bottleneck for global context, and CBAM attention on skip connections.
Multi-scale fusion combines features from all decoder levels before the
segmentation head. Classification uses ROI-guided gating based on predicted
tumor probability with gradient stopping. Deep supervision applies auxiliary
losses at three scales (64×64, 128×128, 256×256). The model has 37M parameters
and processes 4-channel MRI input (FLAIR, T1, T1CE, T2) at 256×256 resolution.}
\label{fig:architecture}
\end{figure*}
```

---

## 🔧 How to Reference in Paper

In methodology section (line ~5), add:
```latex
Figure~\ref{fig:architecture} illustrates the overall architecture.
```

---

## ✅ Checklist Before Finalizing

- [ ] All components labeled clearly
- [ ] Color scheme consistent
- [ ] Arrow directions correct
- [ ] Dimensions shown for each block
- [ ] Skip connections clearly visible
- [ ] Deep supervision branches shown
- [ ] Text is readable (minimum 10pt font)
- [ ] High resolution (300 DPI for PNG, vector for PDF)
- [ ] File saved as `latex/figures/architecture.pdf` or `.png`
- [ ] Figure referenced in methodology.tex
- [ ] Caption added below figure

---

## 🎬 Quick Start with draw.io

1. Go to https://app.diagrams.net/
2. Click "Create New Diagram"
3. Choose "Blank Diagram"
4. Follow the layout above
5. Use these keyboard shortcuts:
   - Ctrl+D: Duplicate selected shape
   - Ctrl+G: Group shapes
   - Alt+Shift+Arrow: Align shapes
6. Export: File → Export as → PDF (or PNG with 300 DPI)
7. Save to `latex/figures/architecture.pdf`

---

## 📚 Alternative: Use These Tools

1. **draw.io** (easiest): https://app.diagrams.net/
2. **PowerPoint/Keynote**: Good for quick diagrams
3. **Inkscape** (free): Professional vector graphics
4. **Adobe Illustrator** (paid): Best quality
5. **Python matplotlib** (programmatic):
   ```python
   import matplotlib.pyplot as plt
   import matplotlib.patches as patches
   # ... draw boxes and arrows
   ```

---

## 🎯 Expected Result

Your final diagram should look like a professional publication figure similar to those in:
- TransUNet (Medical Image Analysis 2021) - Figure 1
- Swin-UNETR (CVPR 2022) - Figure 2
- nnU-Net (Nature Methods 2021) - Figure 1

**Estimated time to create:** 1-2 hours with draw.io

---

**Need help?**
- draw.io tutorial: https://www.diagrams.net/doc/
- Example medical imaging architectures: Search "U-Net architecture diagram" on Google Images
- TikZ examples: https://texample.net/tikz/examples/neural-networks/
