"""Test HD95 computation with realistic data"""
import torch
import numpy as np
from src.braintumnet.metrics import compute_hausdorff_distance_95

def test_hd95():
    print("Testing HD95 computation...")

    # Test 1: Perfect overlap
    pred = np.zeros((128, 128))
    pred[40:80, 40:80] = 1
    target = pred.copy()
    hd95 = compute_hausdorff_distance_95(pred, target)
    print(f"Test 1 (Perfect overlap): HD95 = {hd95:.4f} (expected: 0.0)")

    # Test 2: Slight offset
    pred = np.zeros((128, 128))
    pred[40:80, 40:80] = 1
    target = np.zeros((128, 128))
    target[42:82, 42:82] = 1
    hd95 = compute_hausdorff_distance_95(pred, target)
    print(f"Test 2 (Slight offset): HD95 = {hd95:.4f} (expected: ~2.83)")

    # Test 3: Empty prediction
    pred = np.zeros((128, 128))
    target = np.ones((128, 128))
    hd95 = compute_hausdorff_distance_95(pred, target)
    print(f"Test 3 (Empty pred): HD95 = {hd95} (expected: inf)")

    # Test 4: Both empty
    pred = np.zeros((128, 128))
    target = np.zeros((128, 128))
    hd95 = compute_hausdorff_distance_95(pred, target)
    print(f"Test 4 (Both empty): HD95 = {hd95} (expected: inf)")

    # Test 5: Torch tensor to numpy conversion
    print("\n--- Testing torch tensor conversion ---")
    seg_logits = torch.randn(2, 1, 128, 128) * 5  # Batch of 2
    seg_logits[0, 0, 40:80, 40:80] = 10.0  # High probability region
    seg_logits[1, 0, 50:90, 50:90] = 10.0

    msk = torch.zeros(2, 1, 128, 128)
    msk[0, 0, 42:82, 42:82] = 1.0
    msk[1, 0, 52:92, 52:92] = 1.0

    # Simulate validation loop conversion
    pred_np = (torch.sigmoid(seg_logits) > 0.5).cpu().numpy()
    target_np = msk.cpu().numpy()

    print(f"pred_np shape: {pred_np.shape}, dtype: {pred_np.dtype}")
    print(f"target_np shape: {target_np.shape}, dtype: {target_np.dtype}")
    print(f"pred_np unique: {np.unique(pred_np)}")
    print(f"target_np unique: {np.unique(target_np)}")

    hd95_sum = 0.0
    hd95_count = 0

    for i in range(pred_np.shape[0]):
        pred_slice = pred_np[i, 0] if pred_np.ndim == 4 else pred_np[i]
        target_slice = target_np[i, 0] if target_np.ndim == 4 else target_np[i]

        pred_count = np.sum(pred_slice > 0)
        target_count = np.sum(target_slice > 0)

        print(f"\nSample {i}: pred_pixels={pred_count}, target_pixels={target_count}")

        if pred_count == 0 or target_count == 0:
            print(f"  Skipped (empty mask)")
            continue

        hd95_val = compute_hausdorff_distance_95(pred_slice, target_slice)
        print(f"  HD95 = {hd95_val:.4f}")

        if not np.isinf(hd95_val) and not np.isnan(hd95_val):
            hd95_sum += hd95_val
            hd95_count += 1

    if hd95_count > 0:
        hd95_mean = hd95_sum / hd95_count
        print(f"\nMean HD95: {hd95_mean:.4f} (from {hd95_count} samples)")
    else:
        print(f"\nNo valid HD95 computed! hd95_count = {hd95_count}")
        print("This would result in HD95 = 0.0 in training output!")

if __name__ == "__main__":
    test_hd95()
