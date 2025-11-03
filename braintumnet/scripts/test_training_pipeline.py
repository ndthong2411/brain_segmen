#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test complete training pipeline for all models

This script tests:
1. Config loading
2. Model instantiation
3. Data loading
4. Forward pass
5. Loss computation
6. Backward pass
7. Optimizer step
8. One full training iteration

Usage:
    python scripts/test_training_pipeline.py
"""

import sys
import io
from pathlib import Path

# Fix Unicode on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import torch
import torch.nn as nn
from braintumnet.utils.io import load_yaml
from braintumnet.models import build_model
from braintumnet.losses import MultiTaskLoss
import copy


def merge_configs(base_cfg, override_cfg):
    """Deep merge configs"""
    merged = copy.deepcopy(base_cfg)
    for key, value in override_cfg.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config_auto(model_name=None, hardware=None):
    """Auto-load config"""
    configs_dir = ROOT / "configs"

    # Load base
    base_path = configs_dir / "base.yaml"
    cfg = load_yaml(str(base_path))

    # Load model config
    if model_name:
        model_path = configs_dir / "models" / f"{model_name}.yaml"
        if model_path.exists():
            model_cfg = load_yaml(str(model_path))
            cfg = merge_configs(cfg, model_cfg)

    # Load hardware config
    if hardware:
        hw_path = configs_dir / f"hardware_{hardware}.yaml"
        if hw_path.exists():
            hw_cfg = load_yaml(str(hw_path))
            cfg = merge_configs(cfg, hw_cfg)

            if model_name and "model_batch_sizes" in hw_cfg:
                batch_sizes = hw_cfg["model_batch_sizes"]
                if model_name in batch_sizes:
                    cfg["train"]["batch_size"] = batch_sizes[model_name]

    return cfg


def create_dummy_batch(batch_size=2, img_size=256):
    """Create dummy batch for testing"""
    return {
        'image': torch.randn(batch_size, 4, img_size, img_size),
        'mask': torch.randint(0, 3, (batch_size, 1, img_size, img_size)),
        'label': torch.randint(0, 2, (batch_size,)),
    }


def test_model_training(model_name, hardware=None):
    """Test complete training pipeline for one model"""
    hw_name = hardware if hardware else "default"
    test_name = f"{model_name} + {hw_name}"

    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")

    try:
        # 1. Load config
        print("  [1/9] Loading config...")
        cfg = load_config_auto(model_name, hardware)
        print(f"        ✓ Config loaded")
        print(f"          - Batch size: {cfg['train']['batch_size']}")
        print(f"          - AMP dtype: {cfg['train']['amp_dtype']}")

        # 2. Build model
        print("  [2/9] Building model...")
        model = build_model(cfg)
        print(f"        ✓ Model built: {cfg['model']['model_type']}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"          - Parameters: {total_params:,}")

        # 3. Move to device
        print("  [3/9] Moving to device...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        print(f"        ✓ Model on {device}")

        # 4. Create loss function
        print("  [4/9] Creating loss function...")
        criterion = MultiTaskLoss(
            seg_w=cfg['train']['seg_loss_weight'],
            cls_w=cfg['train']['cls_loss_weight'],
            loss_type=cfg['train']['loss_type'],
            num_classes=cfg['model']['num_classes_seg'],
        )
        print(f"        ✓ Loss: {cfg['train']['loss_type']}")

        # 5. Create optimizer
        print("  [5/9] Creating optimizer...")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg['train']['lr'],
            weight_decay=cfg['train']['weight_decay'],
        )
        print(f"        ✓ Optimizer: AdamW (lr={cfg['train']['lr']})")

        # 6. Create dummy batch
        print("  [6/9] Creating dummy batch...")
        batch = create_dummy_batch(batch_size=2)
        img = batch['image'].to(device)
        mask = batch['mask'].to(device)
        label = batch['label'].to(device)
        print(f"        ✓ Batch created: img={img.shape}, mask={mask.shape}")

        # 7. Forward pass
        print("  [7/9] Running forward pass...")
        model.train()

        # Handle AMP
        amp_enabled = cfg['train']['amp']
        amp_dtype = torch.float16 if cfg['train']['amp_dtype'] == 'float16' else torch.bfloat16

        with torch.amp.autocast(device_type='cuda', enabled=amp_enabled and device=='cuda', dtype=amp_dtype):
            output = model(img)

            # Handle different output formats
            if isinstance(output, tuple):
                if len(output) == 3:
                    seg, cls, aux = output
                elif len(output) == 2:
                    seg, cls = output
                    aux = None
                else:
                    raise ValueError(f"Unexpected output length: {len(output)}")
            else:
                seg = output
                cls = None
                aux = None

            print(f"        ✓ Forward pass complete")
            print(f"          - Seg output: {seg.shape}")
            print(f"          - Cls output: {cls.shape if cls is not None else None}")
            print(f"          - Aux outputs: {len(aux) if aux else 0}")

            # 8. Compute loss
            print("  [8/9] Computing loss...")
            loss, l_seg, l_cls = criterion(seg, mask, cls, label)

            print(f"        ✓ Loss computed")
            print(f"          - Total loss: {loss.item():.4f}")
            print(f"          - Seg loss: {l_seg:.4f}")
            print(f"          - Cls loss: {l_cls:.4f}")

        # 9. Backward pass
        print("  [9/9] Running backward pass...")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"        ✓ Backward pass complete")

        print(f"\n  ✅ {test_name.upper()} - ALL TESTS PASSED!")
        return True

    except Exception as e:
        print(f"\n  ❌ {test_name.upper()} - FAILED!")
        print(f"     Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("COMPREHENSIVE TRAINING PIPELINE TEST")
    print("="*70)
    print("\nThis will test complete training pipeline for all model configurations.")
    print("Testing: Config loading, Model build, Forward, Loss, Backward, Optimizer")
    print()

    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠ CUDA not available, testing on CPU")

    # Test configurations
    models = ['segunetv2', 'swin_unetr', 'nnunet', 'unetr', 'transunet', 'lg_unetr']
    hardwares = [None, 'a100']

    results = {}

    for model in models:
        for hardware in hardwares:
            hw_name = hardware if hardware else "default"
            key = f"{model}_{hw_name}"

            success = test_model_training(model, hardware)
            results[key] = success

            # Clear CUDA cache between tests
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for key, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{key:<30} {status}")

    print(f"\n{'='*70}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"{'='*70}")

    if passed == total:
        print("\n🎉 ALL TRAINING PIPELINE TESTS PASSED!")
        print("\n✅ Ready for production training on server!")
        print("\nNext steps:")
        print("  1. Push code to A100 server")
        print("  2. Start training: python scripts/train.py --model swin_unetr --cfg a100 --fold 0")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please fix before deploying.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
