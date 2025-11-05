# Phase 2 Optional Features

> **Các tính năng tùy chọn trong Phase 2: Multi-Scale Transformer, Attention Gates, Boundary Refinement**

---

## Overview

Phase 2 có **3 optional features** ngoài 7 core improvements:

| Feature | File | Expected Gain | Cost |
|---------|------|---------------|------|
| Multi-Scale Transformer | multiscale_transformer.py | +1.5-2.5% Dice | +40% params bottleneck, +30% time |
| Attention Gates | seg_unet_v2.py (AttentionGate) | +1-2% Dice | +5% params, +10% time |
| Boundary Refinement | seg_unet_v2.py (BoundaryRefinementModule) | +2-3% Dice | +3% params, +5% time |

**Enable**:
```yaml
model:
  use_multiscale_transformer: true   # Optional
  use_attention_gates: true          # Optional
  boundary_refinement: true          # Optional (from Phase 1)
```

---

## 1. Multi-Scale Transformer Bottleneck

### Concept

**Problem**: Single patch size (8×8) → single receptive field
```
Patch 8×8 → Medium receptive field
           → Good for normal features
           → But misses both fine details AND global context
```

**Solution**: Multiple patch sizes (4, 8, 16)
```
Patch 4×4  → Small receptive field  → Fine details
Patch 8×8  → Medium receptive field → Normal features
Patch 16×16 → Large receptive field → Global context
              ↓
           FUSION
              ↓
      Best of all scales!
```

### Architecture

**File**: `multiscale_transformer.py` (243 dòng)

```python
class MultiScaleTransformerBottleneck(nn.Module):
    """
    Multi-Scale Transformer với 3 patch sizes: 4, 8, 16

    Process:
    1. Multi-scale patch embedding (3 scales)
    2. Transformer blocks (shared across scales)
    3. Upsample all to largest resolution
    4. Fuse by concatenation + linear projection
    5. Reshape back to spatial
    """
    def __init__(self, in_ch, dim, patch_sizes=[4, 8, 16],
                 depth=4, n_heads=8):
        # Patch embeddings for each scale
        self.patch_embed = MultiScalePatchEmbed(in_ch, dim, patch_sizes)

        # Shared transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads) for _ in range(depth)
        ])

        # Cross-scale fusion
        self.scale_fusion = nn.Linear(dim * len(patch_sizes), dim)
```

### Forward Pass Detail

```python
def forward(self, x):
    # Input: (B, 512, 32, 32)

    # Step 1: Multi-scale patch embedding
    scale_tokens, scale_shapes = self.patch_embed(x)
    # scale_tokens: [
    #   (B, 64*64=4096, 512),    # Patch 4
    #   (B, 16*16=256, 512),     # Patch 8
    #   (B, 8*8=64, 512)         # Patch 16
    # ]

    # Step 2: Apply transformers to each scale
    processed_scales = []
    for tokens in scale_tokens:
        for block in self.blocks:
            tokens = block(tokens)  # Self-attention + MLP
        processed_scales.append(tokens)

    # Step 3: Upsample all to largest resolution (64×64)
    target_shape = (64, 64)
    upsampled = []
    for tokens, (h, w) in zip(processed_scales, scale_shapes):
        if (h, w) != target_shape:
            # Reshape to spatial → upsample → flatten
            tokens_spatial = tokens.reshape(B, h, w, -1).permute(0, 3, 1, 2)
            tokens_upsampled = F.interpolate(tokens_spatial, size=target_shape)
            tokens = tokens_upsampled.flatten(2).transpose(1, 2)
        upsampled.append(tokens)

    # Step 4: Concatenate and fuse
    fused_tokens = torch.cat(upsampled, dim=-1)  # (B, 4096, 512*3)
    fused_tokens = self.scale_fusion(fused_tokens)  # (B, 4096, 512)

    # Step 5: Reshape back to spatial
    output = fused_tokens.reshape(B, 64, 64, 512).permute(0, 3, 1, 2)
    # → (B, 512, 64, 64)

    # Resize to original input size if needed
    output = F.interpolate(output, size=(32, 32))
    return output  # (B, 512, 32, 32)
```

### What Each Scale Captures

```
Patch 4×4 (64×64 tokens):
  - Fine-grained local patterns
  - Small tumor regions
  - Edge details
  - High spatial resolution

Patch 8×8 (16×16 tokens):
  - Medium-scale structures
  - Typical tumor sizes
  - Standard receptive field
  - Balanced resolution

Patch 16×16 (8×8 tokens):
  - Large-scale context
  - Whole tumor extent
  - Global spatial relationships
  - Low spatial resolution but high semantic
```

### Performance Impact

**Benefits**:
- Better multi-resolution reasoning
- Captures both local and global
- Expected: **+1.5-2.5% Dice**

**Costs**:
- +40% parameters ở bottleneck (512→512×3 then fused)
- +30% slower training
- +2GB memory (multiple token sets)

**When to use**:
- ✅ Có GPU mạnh (A100)
- ✅ Tumors nhiều sizes khác nhau
- ✅ Cần accuracy cao nhất
- ❌ Limited memory (<24GB)
- ❌ Need fast training

**Enable**:
```yaml
# In phase2_a100.yaml
model:
  use_multiscale_transformer: true
  dim: 512                           # Must be large enough
  depth: 4                           # Deep enough to leverage
```

---

## 2. Attention Gates

### Concept

**Problem**: Skip connections có irrelevant features
```
Encoder features (skip) → Contains both useful and noise
                        → Decoder không biết focus vào đâu
                        → Suboptimal feature fusion
```

**Solution**: Attention gate filters skip connections
```
Decoder signal (g) → "What I'm looking for"
Skip features (x)  → "What encoder found"
        ↓
  Attention Gate
        ↓
"Highlight useful, suppress irrelevant"
```

### Architecture

**File**: `seg_unet_v2.py` lines 102-151

```python
class AttentionGate(nn.Module):
    """
    nnU-Net style attention gate

    Args:
        F_g: Channels in gating signal (decoder)
        F_l: Channels in skip connection (encoder)
        F_int: Intermediate channels
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        # Project gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.InstanceNorm2d(F_int, affine=True)
        )

        # Project skip connection
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.InstanceNorm2d(F_int, affine=True)
        )

        # Attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.Sigmoid()
        )

        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: (B, F_g, H, W) gating signal from decoder
            x: (B, F_l, H, W) skip connection from encoder

        Returns:
            (B, F_l, H, W) attention-weighted skip
        """
        # Project both to intermediate dimension
        g1 = self.W_g(g)   # (B, F_int, H, W)
        x1 = self.W_x(x)   # (B, F_int, H, W)

        # Combine: what decoder wants + what encoder has
        combined = self.relu(g1 + x1)  # (B, F_int, H, W)

        # Compute attention coefficients (0-1)
        alpha = self.psi(combined)  # (B, 1, H, W)

        # Apply attention: emphasize useful, suppress irrelevant
        return x * alpha  # (B, F_l, H, W)
```

### How It Works

**Step by step**:

```
1. Decoder tells what to focus on:
   g (decoder): (B, 96, 64, 64) → W_g → (B, 48, 64, 64)

2. Encoder provides features:
   x (skip): (B, 96, 64, 64) → W_x → (B, 48, 64, 64)

3. Combine both signals:
   combined = ReLU(g1 + x1)  # (B, 48, 64, 64)
   "Where decoder's query matches encoder's keys"

4. Compute attention map:
   alpha = Sigmoid(Conv1×1(combined))  # (B, 1, 64, 64)
   Values: 0.0 (irrelevant) to 1.0 (important)

5. Apply attention to skip:
   output = x * alpha  # (B, 96, 64, 64)
   High alpha → keep features
   Low alpha → suppress features
```

### Example

```python
# In decoder level 3
decoder_feat = (B, 96, 64, 64)   # From previous decoder level
skip_feat = (B, 96, 64, 64)      # From encoder level 3

# Without attention gate:
combined = torch.cat([decoder_feat, skip_feat], dim=1)
# → All skip features used equally

# With attention gate:
attn_gate = AttentionGate(F_g=96, F_l=96, F_int=48)
skip_weighted = attn_gate(g=decoder_feat, x=skip_feat)
combined = torch.cat([decoder_feat, skip_weighted], dim=1)
# → Skip features weighted by relevance!
```

### Visualization

```
Attention Map Example:

Original Skip (Tumor + Background + Noise):
┌────────────────┐
│ ████░░░░░░░░░░ │  High activation
│ ████░░░░░▓▓▓░░ │  Tumor region
│ ██████░░░░░░░░ │  + some noise
│ ░░░░░░░░░░░▓▓▓ │  + background
└────────────────┘

Attention Map (alpha):
┌────────────────┐
│ ████░░░░░░░░░░ │  1.0 (keep tumor)
│ ████░░░░░▓░░░░ │  0.7 (keep edge)
│ ██████░░░░░░░░ │  0.3 (suppress background)
│ ░░░░░░░░░░░▓░░ │  0.1 (suppress noise)
└────────────────┘

Weighted Skip (after attention):
┌────────────────┐
│ ████░░░░░░░░░░ │  Tumor emphasized
│ ████░░░░░░░░░░ │  Noise suppressed
│ ██████░░░░░░░░ │  Background reduced
│ ░░░░░░░░░░░░░░ │  Clean features!
└────────────────┘
```

### Performance Impact

**Benefits**:
- Suppress irrelevant regions (background, artifacts)
- Focus on salient features (tumor boundaries)
- Expected: **+1-2% Dice**

**Costs**:
- +5% parameters (attention gate modules)
- +10% slower (extra computations)
- +0.5GB memory (attention maps)

**When to use**:
- ✅ Multi-class với nhiều classes
- ✅ Noisy images
- ✅ Small tumors trong large background
- ✅ Need precise boundaries
- ❌ Memory constrained
- ❌ Already good without it

**Enable**:
```yaml
# In config
model:
  use_attention_gates: true

# All decoder blocks will use attention gates
# d4, d3, d2, d1 → each has AttentionGate
```

---

## 3. Boundary Refinement Module

### Concept

**Problem**: IoU-Dice gap (Dice=0.91, IoU=0.84)
```
Dice: Focuses on overlap
IoU: Penalizes boundary errors more heavily

Gap → Boundary precision not optimal
    → Need explicit boundary modeling
```

**Solution**: Edge detection + boundary attention
```
Features → Edge Detector (Sobel-like)
        → Boundary Attention
        → Refined Features (better edges)
```

### Architecture

**File**: `seg_unet_v2.py` lines 188-248

```python
class BoundaryRefinementModule(nn.Module):
    """
    Boundary refinement for better edge precision

    Components:
    1. Edge detector (initialized with Sobel-like kernels)
    2. Boundary attention mechanism
    3. Multiplicative refinement
    """
    def __init__(self, in_channels):
        super().__init__()

        # Edge detector: depthwise conv initialized with Sobel
        self.edge_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1,
            groups=in_channels,  # Depthwise
            bias=False
        )

        # Initialize with edge detection kernels
        with torch.no_grad():
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            sobel = (sobel_x.abs() + sobel_y.abs()) / 2

            # Apply to all channels
            for i in range(in_channels):
                self.edge_conv.weight[i, 0] = sobel / sobel.sum()

        # Boundary attention: learns to focus on edges
        self.boundary_attn = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.LeakyReLU(0.01),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.LeakyReLU(0.01),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.Sigmoid()  # 0-1 attention weights
        )

    def forward(self, features):
        """
        Args:
            features: (B, C, H, W) decoder output

        Returns:
            refined: (B, C, H, W) with enhanced boundaries
        """
        # Detect edges
        edges = self.edge_conv(features)  # (B, C, H, W)

        # Concatenate features + edges
        combined = torch.cat([features, edges], dim=1)  # (B, 2C, H, W)

        # Generate boundary attention map
        attn = self.boundary_attn(combined)  # (B, C, H, W) ∈ [0, 1]

        # Apply multiplicative attention with residual
        refined = features * (1 + attn)  # Enhance where attn is high

        return refined
```

### How It Works

```
Step 1: Edge Detection
  features: (B, 48, 256, 256)
    ↓ edge_conv (Sobel-like)
  edges: (B, 48, 256, 256)  ← High at boundaries

Step 2: Combine
  combined = [features, edges]  # (B, 96, 256, 256)

Step 3: Boundary Attention
  combined → Conv-Norm-Act → Conv-Norm-Act → Conv-Sigmoid
    ↓
  attn: (B, 48, 256, 256) ∈ [0, 1]
  Where:
    - High near boundaries (e.g., 0.8-1.0)
    - Low in homogeneous regions (e.g., 0.0-0.2)

Step 4: Multiplicative Refinement
  refined = features * (1 + attn)

  Examples:
  - At boundary: attn=0.8 → refined = features * 1.8 (enhanced 80%)
  - In center: attn=0.1 → refined = features * 1.1 (slight boost)
  - Background: attn=0.0 → refined = features * 1.0 (unchanged)
```

### Edge Detection

**Sobel Kernels**:
```
Horizontal edges (Sobel X):
[-1  0  1]
[-2  0  2]
[-1  0  1]

Vertical edges (Sobel Y):
[-1 -2 -1]
[ 0  0  0]
[ 1  2  1]

Combined magnitude:
edge = sqrt(Sobel_X² + Sobel_Y²)
```

**Initialization**:
```python
# Edge conv initialized with Sobel
# But learnable → can adapt to better edge detectors
# After training: may learn different kernels for medical images
```

### Performance Impact

**Benefits**:
- Better edge precision
- Reduces IoU-Dice gap (10% → 5%)
- Expected: **+2-3% Dice**, **+3-4% IoU**

**Costs**:
- +3% parameters (boundary attention network)
- +5% slower (edge detection + attention)
- Minimal memory (+<0.5GB)

**When to use**:
- ✅ Large IoU-Dice gap
- ✅ Need precise boundaries (clinical)
- ✅ Small tumors (boundaries matter more)
- ✅ Post-processing alternative
- ✅ Always recommended (low cost)

**Enable**:
```yaml
# In config
model:
  boundary_refinement: true  # From Phase 1 optimization

# Applied after final decoder output (d1)
# Before segmentation head
```

---

## Feature Combinations

### Recommended Combinations

**Budget Setup** (RTX 3060 12GB):
```yaml
# None or only boundary refinement
use_multiscale_transformer: false
use_attention_gates: false
boundary_refinement: true      # Low cost, good benefit
```

**Balanced Setup** (RTX 3090 24GB):
```yaml
# Boundary + one advanced feature
use_multiscale_transformer: false
use_attention_gates: true      # OR multiscale, not both
boundary_refinement: true
```

**High-End Setup** (A100 80GB):
```yaml
# All features
use_multiscale_transformer: true
use_attention_gates: true
boundary_refinement: true
```

### Expected Cumulative Gains

```
Baseline V2 (no optional features):
  Dice: 0.91-0.92

+ Boundary Refinement:
  Dice: 0.93-0.94 (+2-3%)

+ Attention Gates:
  Dice: 0.94-0.95 (+1-2% more)

+ Multi-Scale Transformer:
  Dice: 0.95-0.96 (+1-2% more)

Total possible gain: +4-7% over baseline V2
```

**Note**: Gains are NOT strictly additive, expect some overlap

---

## Implementation Notes

### Enable/Disable Features

```python
# In model initialization
model = SegUNetV2(
    # ... other params ...

    # Phase 2 optional features
    use_multiscale_transformer=True,   # Multi-scale transformer
    use_attention_gates=True,          # Attention gates
    boundary_refinement=True,          # Boundary refinement
)

# All features default to False for backward compatibility
```

### Training Tips

**With Multi-Scale Transformer**:
- Increase warmup steps (1000 → 2000)
- Lower learning rate (5e-5 → 3e-5)
- More epochs (350 → 400)
- Larger GPU needed

**With Attention Gates**:
- Standard training works
- May need slightly lower LR
- Monitor validation carefully (can overfit)

**With Boundary Refinement**:
- No special training needed
- Works with standard config
- Minimal overhead

---

## Summary

### Feature Comparison

| Feature | Benefit | Cost | Recommended For |
|---------|---------|------|-----------------|
| **Multi-Scale Transformer** | +1.5-2.5% Dice | High (+40% params, +30% time) | A100, max accuracy |
| **Attention Gates** | +1-2% Dice | Medium (+5% params, +10% time) | Multi-class, noisy data |
| **Boundary Refinement** | +2-3% Dice | Low (+3% params, +5% time) | Always (unless memory critical) |

### When to Use What

**Always use**:
- ✅ Boundary Refinement (low cost, good benefit)

**Use if have GPU**:
- ✅ Attention Gates (RTX 3090+)
- ✅ Multi-Scale Transformer (A100)

**Skip if**:
- ❌ Limited memory (<20GB)
- ❌ Need fast training
- ❌ Already good results

---

**Next**: [Training Configuration →](v2_04_TRAINING_CONFIG.md)

**Back**: [← Architecture](v2_02_SEGUNETV2_ARCHITECTURE.md)
