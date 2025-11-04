#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test all SOTA models with dummy data

This script tests:
1. Model instantiation
2. Forward pass
3. Output shapes
4. Parameter counts

Usage:
    python scripts/test_models.py
"""

import sys
import os
from pathlib import Path

# Fix Unicode output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import yaml


def test_model(model_type, config_path):
    """Test a single model"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_type.upper()}")
    print(f"{'='*60}")

    try:
        # Load config
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Override model type
        cfg['model']['model_type'] = model_type

        # Build model
        from braintumnet.models import build_model
        model = build_model(cfg)

        print(f"✓ Model instantiated successfully")

        # Create dummy input
        batch_size = 2
        x = torch.randn(batch_size, 4, 256, 256)
        print(f"  Input shape: {x.shape}")

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x)

        # Handle different output formats
        if isinstance(output, tuple):
            if len(output) == 3:
                seg_logits, cls_logits, aux_outputs = output
            elif len(output) == 2:
                seg_logits, cls_logits = output
                aux_outputs = None
            else:
                raise ValueError(f"Unexpected output length: {len(output)}")
        else:
            seg_logits = output
            cls_logits = None
            aux_outputs = None

        print(f"✓ Forward pass successful")
        print(f"  Segmentation output: {seg_logits.shape}")
        print(f"  Classification output: {cls_logits}")

        if aux_outputs is not None:
            print(f"  Auxiliary outputs: {[aux.shape for aux in aux_outputs]}")

        # Check output shape
        expected_shape = (batch_size, 3, 256, 256)
        if seg_logits.shape != expected_shape:
            raise ValueError(f"Wrong output shape! Expected {expected_shape}, got {seg_logits.shape}")

        print(f"✓ Output shape correct: {seg_logits.shape}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        print(f"✓ Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

        print(f"\n✅ {model_type.upper()} TEST PASSED!")
        return True

    except Exception as e:
        print(f"\n❌ {model_type.upper()} TEST FAILED!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("Testing SOTA Models")
    print("="*60)

    # Base config path
    base_config = ROOT / "configs" / "phase2_a100_lmdb.yaml"

    # Models to test
    models_to_test = [
        ("segunetv2", base_config),
        ("swin_unetr", ROOT / "configs" / "model_swin_unetr.yaml"),
        ("nnunet", ROOT / "configs" / "model_nnunet.yaml"),
        ("unetr", ROOT / "configs" / "model_unetr.yaml"),
    ]

    results = {}

    for model_type, config_path in models_to_test:
        success = test_model(model_type, config_path)
        results[model_type] = success

    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    for model_type, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{model_type:<20} {status}")

    # Overall result
    all_passed = all(results.values())
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
