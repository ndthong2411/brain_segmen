"""
Test script to verify Deep Supervision and Boundary Loss implementation.

Usage:
    python scripts/test_improvements.py
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, 'src')

from braintumnet.models.braintumnet import BrainTumNet
from braintumnet.losses import MultiTaskLoss, BoundaryLoss, dice_loss_with_logits

def test_deep_supervision():
    """Test deep supervision implementation."""
    print("\n" + "="*60)
    print("TEST 1: Deep Supervision")
    print("="*60)

    # Create model with deep supervision
    model = BrainTumNet(
        in_ch=4,
        num_cls=2,
        base=32,
        dim=256,
        patch=8,
        depth=2,
        n_heads=4,
        deep_supervision=True  # Enable deep supervision
    )

    print(f"✓ Model created with deep_supervision=True")

    # Create dummy input
    batch_size = 2
    img = torch.randn(batch_size, 4, 256, 256)

    # Forward pass
    output = model(img)

    # Check output format
    if len(output) == 3:
        seg_logits, cls_logits, aux_outputs = output
        print(f"✓ Model returns 3 outputs (seg, cls, aux)")

        # Check main outputs
        assert seg_logits.shape == (batch_size, 1, 256, 256), f"Wrong seg shape: {seg_logits.shape}"
        assert cls_logits.shape == (batch_size, 2), f"Wrong cls shape: {cls_logits.shape}"
        print(f"✓ Main outputs: seg {seg_logits.shape}, cls {cls_logits.shape}")

        # Check auxiliary outputs
        assert len(aux_outputs) == 3, f"Expected 3 aux outputs, got {len(aux_outputs)}"
        aux3, aux2, aux1 = aux_outputs

        assert aux3.shape == (batch_size, 1, 64, 64), f"Wrong aux3 shape: {aux3.shape}"
        assert aux2.shape == (batch_size, 1, 128, 128), f"Wrong aux2 shape: {aux2.shape}"
        assert aux1.shape == (batch_size, 1, 256, 256), f"Wrong aux1 shape: {aux1.shape}"
        print(f"✓ Auxiliary outputs:")
        print(f"  - aux3: {aux3.shape} (64×64)")
        print(f"  - aux2: {aux2.shape} (128×128)")
        print(f"  - aux1: {aux1.shape} (256×256)")

        print(f"\n✅ Deep Supervision test PASSED")
        return True
    else:
        print(f"❌ Expected 3 outputs, got {len(output)}")
        return False


def test_deep_supervision_disabled():
    """Test model with deep supervision disabled."""
    print("\n" + "="*60)
    print("TEST 2: Deep Supervision Disabled (Backward Compatibility)")
    print("="*60)

    # Create model WITHOUT deep supervision
    model = BrainTumNet(
        in_ch=4,
        num_cls=2,
        base=32,
        deep_supervision=False  # Disabled
    )

    print(f"✓ Model created with deep_supervision=False")

    # Create dummy input
    batch_size = 2
    img = torch.randn(batch_size, 4, 256, 256)

    # Forward pass
    output = model(img)

    # Check output format (should be 2 outputs only)
    if len(output) == 2:
        seg_logits, cls_logits = output
        print(f"✓ Model returns 2 outputs (seg, cls) only")
        print(f"  - seg: {seg_logits.shape}")
        print(f"  - cls: {cls_logits.shape}")
        print(f"\n✅ Backward compatibility test PASSED")
        return True
    else:
        print(f"❌ Expected 2 outputs, got {len(output)}")
        return False


def test_boundary_loss():
    """Test boundary loss implementation."""
    print("\n" + "="*60)
    print("TEST 3: Boundary Loss")
    print("="*60)

    # Create boundary loss
    boundary_loss_fn = BoundaryLoss(cache_distance_maps=True)
    print(f"✓ BoundaryLoss created with caching enabled")

    # Create dummy data
    batch_size = 2
    H, W = 256, 256

    # Create synthetic mask (circle in center)
    y, x = np.ogrid[-H//2:H//2, -W//2:W//2]
    mask_np = (x*x + y*y <= 50*50).astype(np.float32)
    mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # Create predictions (slightly shifted)
    pred_logits = torch.randn(batch_size, 1, H, W) * 2.0

    print(f"✓ Created synthetic data:")
    print(f"  - mask: {mask.shape}, values in {mask.min():.2f} to {mask.max():.2f}")
    print(f"  - pred_logits: {pred_logits.shape}")

    # Compute loss
    try:
        loss = boundary_loss_fn(pred_logits, mask)
        print(f"✓ Boundary loss computed: {loss.item():.6f}")

        # Check cache
        cache_size = len(boundary_loss_fn.cache)
        print(f"✓ Distance map cache size: {cache_size}")
        assert cache_size > 0, "Cache should contain at least one entry"

        # Test cache hit
        loss2 = boundary_loss_fn(pred_logits, mask)
        print(f"✓ Cache hit test: {loss2.item():.6f} (should be same)")
        assert abs(loss.item() - loss2.item()) < 1e-6, "Cache should return same value"

        print(f"\n✅ Boundary Loss test PASSED")
        return True

    except Exception as e:
        print(f"❌ Boundary loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multitask_loss_with_boundary():
    """Test MultiTaskLoss with boundary loss."""
    print("\n" + "="*60)
    print("TEST 4: MultiTaskLoss with Boundary Loss")
    print("="*60)

    # Create loss without boundary
    loss_no_boundary = MultiTaskLoss(seg_w=1.0, cls_w=0.5, boundary_w=0.0)
    print(f"✓ MultiTaskLoss created WITHOUT boundary (weight=0.0)")
    print(f"  - boundary_loss initialized: {loss_no_boundary.boundary_loss is not None}")

    # Create loss with boundary
    loss_with_boundary = MultiTaskLoss(seg_w=1.0, cls_w=0.5, boundary_w=0.2)
    print(f"✓ MultiTaskLoss created WITH boundary (weight=0.2)")
    print(f"  - boundary_loss initialized: {loss_with_boundary.boundary_loss is not None}")

    # Create dummy data
    batch_size = 2
    seg_logits = torch.randn(batch_size, 1, 256, 256)
    seg_mask = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    cls_logits = torch.randn(batch_size, 2)
    cls_label = torch.randint(0, 2, (batch_size,))

    # Compute losses
    try:
        loss1, l_seg1, l_cls1 = loss_no_boundary(seg_logits, seg_mask, cls_logits, cls_label)
        print(f"✓ Loss WITHOUT boundary: total={loss1.item():.4f}, seg={l_seg1.item():.4f}, cls={l_cls1.item():.4f}")

        loss2, l_seg2, l_cls2 = loss_with_boundary(seg_logits, seg_mask, cls_logits, cls_label)
        print(f"✓ Loss WITH boundary: total={loss2.item():.4f}, seg={l_seg2.item():.4f}, cls={l_cls2.item():.4f}")

        # Loss with boundary should be higher (boundary term added)
        assert loss2 > loss1, f"Boundary loss should increase total loss: {loss2.item()} vs {loss1.item()}"
        print(f"✓ Boundary loss increases total loss as expected")

        print(f"\n✅ MultiTaskLoss test PASSED")
        return True

    except Exception as e:
        print(f"❌ MultiTaskLoss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_auxiliary_loss_computation():
    """Test auxiliary loss computation logic."""
    print("\n" + "="*60)
    print("TEST 5: Auxiliary Loss Computation")
    print("="*60)

    # Simulate auxiliary outputs
    batch_size = 2
    aux3 = torch.randn(batch_size, 1, 64, 64)    # Coarsest
    aux2 = torch.randn(batch_size, 1, 128, 128)
    aux1 = torch.randn(batch_size, 1, 256, 256)  # Finest
    mask = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()

    aux_outputs = [aux3, aux2, aux1]
    aux_weights = [0.5, 0.25, 0.125]

    print(f"✓ Created auxiliary outputs:")
    for i, (aux, weight) in enumerate(zip(aux_outputs, aux_weights)):
        print(f"  - aux{i+1}: {aux.shape}, weight={weight}")

    # Compute auxiliary losses
    total_aux_loss = 0.0
    try:
        for i, aux_output in enumerate(aux_outputs):
            # Resize to match mask
            aux_resized = F.interpolate(aux_output, size=mask.shape[-2:],
                                       mode='bilinear', align_corners=False)
            assert aux_resized.shape == mask.shape, f"Resize failed: {aux_resized.shape} vs {mask.shape}"

            # Compute loss
            aux_loss = dice_loss_with_logits(aux_resized, mask) + \
                      F.binary_cross_entropy_with_logits(aux_resized, mask)

            weight = aux_weights[i]
            weighted_loss = weight * aux_loss
            total_aux_loss += weighted_loss

            print(f"✓ aux{i+1}: loss={aux_loss.item():.4f}, weighted={weighted_loss.item():.4f}")

        print(f"✓ Total auxiliary loss: {total_aux_loss.item():.4f}")
        print(f"\n✅ Auxiliary loss computation test PASSED")
        return True

    except Exception as e:
        print(f"❌ Auxiliary loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TESTING DEEP SUPERVISION + BOUNDARY LOSS IMPLEMENTATION")
    print("="*60)

    results = []

    # Run tests
    results.append(("Deep Supervision", test_deep_supervision()))
    results.append(("Deep Supervision Disabled", test_deep_supervision_disabled()))
    results.append(("Boundary Loss", test_boundary_loss()))
    results.append(("MultiTaskLoss with Boundary", test_multitask_loss_with_boundary()))
    results.append(("Auxiliary Loss Computation", test_auxiliary_loss_computation()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:40s} {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYou can now train with:")
        print("  python scripts/train.py --config configs/improved_v1_deep_supervision.yaml --fold 0 --epochs 5")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Please fix before training")
        return 1


if __name__ == "__main__":
    exit(main())
