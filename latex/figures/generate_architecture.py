"""
Generate AMT-UNet Architecture Diagram
Uses matplotlib to create publication-quality figure
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Set style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9

# Create figure
fig = plt.figure(figsize=(14, 18))
ax = fig.add_subplot(111)
ax.set_xlim(0, 14)
ax.set_ylim(0, 18)
ax.axis('off')

# Colors
COLOR_INPUT = '#F0F0F0'
COLOR_ENCODER = '#ADD8E6'
COLOR_DECODER = '#90EE90'
COLOR_TRANSFORMER = '#FFD700'
COLOR_FUSION = '#DDA0DD'
COLOR_SEG = '#4169E1'
COLOR_CLS = '#DC143C'
COLOR_AUX = '#FFA500'
COLOR_BORDER = '#404040'

def add_box(ax, x, y, width, height, text, color, fontsize=9, fontweight='normal'):
    """Add a box with text"""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.05",
        edgecolor=COLOR_BORDER,
        facecolor=color,
        linewidth=2
    )
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight,
            multialignment='center')
    return box

def add_arrow(ax, x1, y1, x2, y2, style='solid', width=2, color='black', label=''):
    """Add arrow between boxes"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->' if style == 'solid' else '->',
        linestyle='-' if style == 'solid' else '--',
        linewidth=width,
        color=color,
        mutation_scale=20,
        zorder=1
    )
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, label, fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'),
                ha='center', va='center')

# ============================================================
# INPUT
# ============================================================
add_box(ax, 5, 17, 4, 0.7,
        'INPUT\n4-channel MRI (FLAIR, T1, T1CE, T2)\n256×256×4',
        COLOR_INPUT, fontsize=10, fontweight='bold')

# Arrow down
add_arrow(ax, 7, 17, 7, 16.2, width=3)

# ============================================================
# ENCODER (Left side)
# ============================================================
encoder_x = 0.5
encoder_y_start = 12
encoder_width = 2.5
encoder_height = 0.9
encoder_spacing = 1.1

# Encoder blocks
encoders = [
    ('E1\nResConvBlock\n48 ch\n256×256', 16.2),
    ('E2\nResConvBlock\n96 ch\n128×128', 14.8),
    ('E3\nResConvBlock\n192 ch\n64×64', 13.4),
    ('E4\nResConvBlock\n384 ch\n32×32', 12.0),
]

encoder_centers = []
for i, (text, y) in enumerate(encoders):
    add_box(ax, encoder_x, y, encoder_width, encoder_height, text, COLOR_ENCODER, fontsize=8)
    encoder_centers.append((encoder_x + encoder_width/2, y + encoder_height/2))

    # Arrow down to next (except last)
    if i < len(encoders) - 1:
        add_arrow(ax, encoder_x + encoder_width/2, y,
                 encoder_x + encoder_width/2, y - 0.5, width=2)
        # Label stride
        ax.text(encoder_x + encoder_width/2 + 0.3, y - 0.25, 'stride 2',
                fontsize=7, style='italic', color='gray')

# Arrow to transformer
add_arrow(ax, encoder_x + encoder_width/2, 12,
         encoder_x + encoder_width/2, 11, width=2)

# ============================================================
# TRANSFORMER (Center bottom)
# ============================================================
trans_x = 3.5
trans_y = 8.5
trans_width = 4.5
trans_height = 2.5

trans_box = FancyBboxPatch(
    (trans_x, trans_y), trans_width, trans_height,
    boxstyle="round,pad=0.1",
    edgecolor=COLOR_BORDER,
    facecolor=COLOR_TRANSFORMER,
    linewidth=3,
    zorder=2
)
ax.add_patch(trans_box)

trans_text = '''ADAPTIVE MASKED TRANSFORMER
--------------------
Patch Embed (8x8) -> 4 tokens
Soft Mask Generator (MLP)

Masked Self-Attention
(4 layers, 8 heads, dim=384)
--------------------
Output: 384x16x16'''

ax.text(trans_x + trans_width/2, trans_y + trans_height/2, trans_text,
        ha='center', va='center', fontsize=8, fontweight='bold',
        multialignment='center')

# Arrows: encoder -> transformer
add_arrow(ax, encoder_x + encoder_width/2, 11, trans_x + trans_width/2, trans_y + trans_height, width=2)

# ============================================================
# DECODER (Right side)
# ============================================================
decoder_x = 11
decoder_y_start = 12
decoder_width = 2.5
decoder_height = 0.9

decoders = [
    ('D4\nUpConv+CBAM\n384 ch\n32×32', 12.0),
    ('D3\nUpConv+CBAM\n192 ch\n64×64', 13.4),
    ('D2\nUpConv+CBAM\n96 ch\n128×128', 14.8),
    ('D1\nUpConv+CBAM\n48 ch\n256×256', 16.2),
]

decoder_centers = []
for i, (text, y) in enumerate(decoders):
    add_box(ax, decoder_x, y, decoder_width, decoder_height, text, COLOR_DECODER, fontsize=8)
    decoder_centers.append((decoder_x + decoder_width/2, y + decoder_height/2))

    # Arrow up to next (except last)
    if i < len(decoders) - 1:
        add_arrow(ax, decoder_x + decoder_width/2, y + decoder_height,
                 decoder_x + decoder_width/2, decoders[i+1][1], width=2)

# Arrow from transformer to first decoder
add_arrow(ax, trans_x + trans_width/2, trans_y + trans_height,
         decoder_x + decoder_width/2, 12, width=2)

# ============================================================
# SKIP CONNECTIONS (dashed arrows with CBAM label)
# ============================================================
for i in range(len(encoders)):
    enc_x, enc_y = encoder_centers[i]
    dec_x, dec_y = decoder_centers[len(decoders) - 1 - i]

    # Curved arrow through center
    mid_x = 7
    mid_y = (enc_y + dec_y) / 2

    # Draw dashed curve
    add_arrow(ax, enc_x + encoder_width/2, enc_y, mid_x, mid_y,
             style='dashed', width=1.5, color='gray', label='CBAM')
    add_arrow(ax, mid_x, mid_y, dec_x - decoder_width/2, dec_y,
             style='dashed', width=1.5, color='gray')

# ============================================================
# DEEP SUPERVISION (Auxiliary outputs)
# ============================================================
aux_x = 13.8
aux_outputs = [
    ('Aux3\n64×64', 12.3, decoder_centers[0]),
    ('Aux2\n128×128', 13.7, decoder_centers[1]),
    ('Aux1\n256×256', 15.1, decoder_centers[2]),
]

for text, y, (dec_x, dec_y) in aux_outputs:
    add_box(ax, aux_x, y, 0.8, 0.5, text, COLOR_AUX, fontsize=7)
    add_arrow(ax, dec_x + decoder_width/2, dec_y, aux_x, y + 0.25,
             style='dashed', width=1, color=COLOR_AUX)

# ============================================================
# MULTI-SCALE FUSION
# ============================================================
fusion_x = 8.5
fusion_y = 10.8
fusion_width = 3
fusion_height = 0.8

add_box(ax, fusion_x, fusion_y, fusion_width, fusion_height,
        'MULTI-SCALE FUSION\nInterpolate all D levels → Sum → Normalize',
        COLOR_FUSION, fontsize=8, fontweight='bold')

# Arrow from D1 to fusion
add_arrow(ax, decoder_x + decoder_width/2, 16.2,
         fusion_x + fusion_width/2, fusion_y + fusion_height, width=2)

# ============================================================
# SEGMENTATION HEAD
# ============================================================
seg_x = 8.5
seg_y = 9.5
seg_width = 3
seg_height = 0.8

add_box(ax, seg_x, seg_y, seg_width, seg_height,
        'SEGMENTATION HEAD\nResConvBlock + Conv1×1\n→ 3 classes (BG, TC, ED)',
        COLOR_SEG, fontsize=8, fontweight='bold')

add_arrow(ax, fusion_x + fusion_width/2, fusion_y,
         seg_x + seg_width/2, seg_y + seg_height, width=2)

# Output
add_box(ax, 8.5, 8.5, 3, 0.6,
        'Segmentation Output\n256×256×3',
        COLOR_SEG, fontsize=9, fontweight='bold')

add_arrow(ax, seg_x + seg_width/2, seg_y,
         seg_x + seg_width/2, 9.1, width=2)

# ============================================================
# CLASSIFICATION BRANCH
# ============================================================
# ROI Gating
roi_x = 3
roi_y = 6.5
roi_width = 2.5
roi_height = 0.9

add_box(ax, roi_x, roi_y, roi_width, roi_height,
        'ROI GATING\nConv1×1 (4→1)\n× WT prob (detach grad)',
        COLOR_CLS, fontsize=8, fontweight='bold')

# Arrow from input
add_arrow(ax, 5, 17, roi_x + roi_width/2, roi_y + roi_height, width=1.5, color=COLOR_CLS)

# Arrow from segmentation (dotted)
add_arrow(ax, seg_x, seg_y + seg_height/2, roi_x + roi_width, roi_y + roi_height/2,
         style='dashed', width=1, color=COLOR_CLS, label='WT prob')

# Classification Network
cls_x = 3
cls_y = 4.5
cls_width = 2.5
cls_height = 1.5

cls_box = FancyBboxPatch(
    (cls_x, cls_y), cls_width, cls_height,
    boxstyle="round,pad=0.1",
    edgecolor=COLOR_BORDER,
    facecolor=COLOR_CLS,
    linewidth=2
)
ax.add_patch(cls_box)

cls_text = '''CLASSIFICATION NET
T-Inception
64->128->256
Global Pool
Dropout (0.3)
-> 2 classes'''

ax.text(cls_x + cls_width/2, cls_y + cls_height/2, cls_text,
        ha='center', va='center', fontsize=8, fontweight='bold',
        multialignment='center', color='white')

add_arrow(ax, roi_x + roi_width/2, roi_y,
         cls_x + cls_width/2, cls_y + cls_height, width=2, color=COLOR_CLS)

# Output
add_box(ax, 3, 3.5, 2.5, 0.6,
        'Classification Output\nHGG / LGG',
        COLOR_CLS, fontsize=9, fontweight='bold')

add_arrow(ax, cls_x + cls_width/2, cls_y,
         cls_x + cls_width/2, 4.1, width=2, color=COLOR_CLS)

# ============================================================
# LEGEND / KEY FEATURES
# ============================================================
legend_x = 0.5
legend_y = 0.5
legend_width = 13
legend_height = 2.5

legend_box = FancyBboxPatch(
    (legend_x, legend_y), legend_width, legend_height,
    boxstyle="round,pad=0.1",
    edgecolor=COLOR_BORDER,
    facecolor='#FFFACD',
    linewidth=2
)
ax.add_patch(legend_box)

legend_text = '''KEY FEATURES:
- Instance Normalization (batch-size independent)           - CBAM Attention on Skip Connections
- Residual Connections (gradient flow)                      - Multi-Scale Fusion (all decoder levels)
- Strided Convolution Downsampling (learned)                - ROI-Guided Classification (gradient stopping)
- Adaptive Masked Transformer (4 layers, 8 heads, 384 dim) - Deep Supervision (weight 0.3 at 3 scales)

TOTAL PARAMETERS: 37M  |  INPUT: 256x256x4  |  OUTPUTS: Segmentation (3 classes) + Classification (2 classes)'''

ax.text(legend_x + legend_width/2, legend_y + legend_height/2, legend_text,
        ha='center', va='center', fontsize=8,
        multialignment='left')

# ============================================================
# TITLE
# ============================================================
ax.text(7, 17.8, 'AMT-UNet Architecture',
        ha='center', va='center', fontsize=14, fontweight='bold')

# ============================================================
# Save
# ============================================================
plt.tight_layout()
plt.savefig('architecture.pdf', bbox_inches='tight', dpi=300)
plt.savefig('architecture.png', bbox_inches='tight', dpi=300)
print("Architecture diagram saved as:")
print("  - architecture.pdf (vector, publication quality)")
print("  - architecture.png (raster, 300 DPI)")
plt.close()
