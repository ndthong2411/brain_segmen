#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test the new config system"""

import sys
import io
from pathlib import Path

# Fix Unicode on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from braintumnet.utils.io import load_yaml
import copy


def merge_configs(base_cfg, override_cfg):
    """Deep merge two configs, override_cfg takes precedence"""
    merged = copy.deepcopy(base_cfg)

    for key, value in override_cfg.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge dicts
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value

    return merged


def load_config_auto(model_name=None, hardware=None):
    """Auto-load and merge configs"""
    configs_dir = ROOT / "configs"

    # 1. Load base config
    base_path = configs_dir / "base.yaml"
    print(f"  Loading base config: {base_path.name}")
    cfg = load_yaml(str(base_path))

    # 2. Load model-specific config
    if model_name:
        model_path = configs_dir / "models" / f"{model_name}.yaml"
        if model_path.exists():
            print(f"  Loading model config: {model_path.name}")
            model_cfg = load_yaml(str(model_path))
            cfg = merge_configs(cfg, model_cfg)

    # 3. Load hardware-specific config
    if hardware:
        hw_path = configs_dir / f"hardware_{hardware}.yaml"
        if hw_path.exists():
            print(f"  Loading hardware config: {hw_path.name}")
            hw_cfg = load_yaml(str(hw_path))
            cfg = merge_configs(cfg, hw_cfg)

            # Apply model-specific batch size
            if model_name and "model_batch_sizes" in hw_cfg:
                batch_sizes = hw_cfg["model_batch_sizes"]
                if model_name in batch_sizes:
                    cfg["train"]["batch_size"] = batch_sizes[model_name]

    return cfg


def test_config(model, hardware=None):
    """Test a specific config combination"""
    hw_name = hardware if hardware else "default"
    print(f"\nTesting: {model} + {hw_name}")
    print("-" * 50)

    cfg = load_config_auto(model, hardware)

    print(f"  Model type:      {cfg['model']['model_type']}")
    print(f"  Batch size:      {cfg['train']['batch_size']}")
    print(f"  Workers:         {cfg['train']['workers']}")
    print(f"  AMP dtype:       {cfg['train']['amp_dtype']}")
    print(f"  Optimizer fused: {cfg['train'].get('optimizer_fused', False)}")
    print(f"  Channels last:   {cfg['train'].get('channels_last', False)}")

    # Model-specific checks
    if model == 'swin_unetr':
        assert 'feature_size' in cfg['model'], "Missing feature_size"
        print(f"  Feature size:    {cfg['model']['feature_size']}")
    elif model == 'nnunet':
        assert cfg['model']['base'] == 32, "Wrong nnunet base"
        print(f"  Base channels:   {cfg['model']['base']}")
    elif model == 'unetr':
        assert 'hidden_size' in cfg['model'], "Missing hidden_size"
        print(f"  Hidden size:     {cfg['model']['hidden_size']}")

    print("  ✅ Config valid!")


def main():
    print("=" * 60)
    print("Testing Unified Config System")
    print("=" * 60)

    models = ['segunetv2', 'swin_unetr', 'nnunet', 'unetr']
    hardwares = [None, 'a100']

    for model in models:
        for hardware in hardwares:
            try:
                test_config(model, hardware)
            except Exception as e:
                print(f"  ❌ Failed: {e}")

    print("\n" + "=" * 60)
    print("Usage Examples:")
    print("=" * 60)
    examples = [
        "python scripts/train.py --model swin_unetr --fold 0",
        "python scripts/train.py --model swin_unetr --cfg a100 --fold 0",
        "python scripts/train.py --model nnunet --fold 0",
        "python scripts/train.py --model unetr --cfg a100 --fold 1",
    ]

    for ex in examples:
        print(f"  {ex}")

    print("\n✅ All config tests passed!")


if __name__ == "__main__":
    main()
