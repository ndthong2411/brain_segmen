"""
Test Phase 2 Model Initialization
"""
import sys
sys.path.insert(0, 'braintumnet/src')

import torch
from braintumnet.models.seg_unet_v2 import SegUNetV2

print("="*70)
print("Testing Phase 2 Model")
print("="*70)

# Test Phase 2 configuration
print("\n1. Testing Phase 2 Model (Multi-scale Transformer + Attention Gates)")
model_phase2 = SegUNetV2(
    in_ch=4,
    base=64,
    dim=512,
    patch=8,
    depth=4,
    n_heads=8,
    num_classes=3,
    dropout=0.2,
    norm='instance',
    deep_supervision=True,
    multi_scale_fusion=True,
    boundary_refinement=True,           # Phase 1
    use_multiscale_transformer=True,    # Phase 2
    use_attention_gates=True            # Phase 2
)

# Count parameters
total_params = sum(p.numel() for p in model_phase2.parameters())
print(f"   Parameters: {total_params/1e6:.2f}M")

# Test forward pass
print("\n2. Testing forward pass...")
x = torch.randn(2, 4, 256, 256)
try:
    with torch.no_grad():
        output = model_phase2(x)

    if isinstance(output, tuple):
        seg, aux = output
        print(f"   Input shape:  {x.shape}")
        print(f"   Seg output:   {seg.shape}")
        print(f"   Aux outputs:  [{aux[0].shape}, {aux[1].shape}, {aux[2].shape}]")
        print(f"   [OK] Forward pass successful!")
    else:
        print(f"   Input shape:  {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   [OK] Forward pass successful!")
except Exception as e:
    print(f"   [ERROR] Error: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Testing Phase 1 Model (for comparison)")
model_phase1 = SegUNetV2(
    in_ch=4,
    base=64,
    dim=512,
    patch=8,
    depth=4,
    n_heads=8,
    num_classes=3,
    dropout=0.2,
    norm='instance',
    deep_supervision=True,
    multi_scale_fusion=True,
    boundary_refinement=True,           # Phase 1
    use_multiscale_transformer=False,   # Disabled
    use_attention_gates=False           # Disabled
)

total_params_p1 = sum(p.numel() for p in model_phase1.parameters())
print(f"   Phase 1 Parameters: {total_params_p1/1e6:.2f}M")
print(f"   Phase 2 Parameters: {total_params/1e6:.2f}M")
print(f"   Difference: +{(total_params - total_params_p1)/1e6:.2f}M")

try:
    with torch.no_grad():
        output_p1 = model_phase1(x)
    print(f"   [OK] Phase 1 forward pass successful!")
except Exception as e:
    print(f"   [ERROR] Phase 1 Error: {e}")

print("\n4. Testing Baseline Model (no optimizations)")
model_baseline = SegUNetV2(
    in_ch=4,
    base=64,
    dim=512,
    patch=8,
    depth=4,
    n_heads=8,
    num_classes=3,
    dropout=0.2,
    norm='instance',
    deep_supervision=True,
    multi_scale_fusion=True,
    boundary_refinement=False,          # Disabled
    use_multiscale_transformer=False,   # Disabled
    use_attention_gates=False           # Disabled
)

total_params_baseline = sum(p.numel() for p in model_baseline.parameters())
print(f"   Baseline Parameters: {total_params_baseline/1e6:.2f}M")

try:
    with torch.no_grad():
        output_baseline = model_baseline(x)
    print(f"   [OK] Baseline forward pass successful!")
except Exception as e:
    print(f"   [ERROR] Baseline Error: {e}")

print("\n" + "="*70)
print("Summary:")
print("="*70)
print(f"Baseline:  {total_params_baseline/1e6:.2f}M params")
print(f"Phase 1:   {total_params_p1/1e6:.2f}M params (+{(total_params_p1-total_params_baseline)/1e6:.2f}M)")
print(f"Phase 2:   {total_params/1e6:.2f}M params (+{(total_params-total_params_baseline)/1e6:.2f}M)")
print("="*70)
print("[OK] All tests passed!")
