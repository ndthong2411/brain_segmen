#!/usr/bin/env python3
"""
Compare Different Training Configurations

Shows side-by-side comparison of Phase 1, Phase 2 Small, and Phase 2 A100.
"""

import yaml
from pathlib import Path
from typing import Dict


def load_config(config_path: str) -> Dict:
    """Load YAML config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_config_info(config: Dict, name: str) -> Dict:
    """Extract key info from config."""
    model = config.get('model', {})
    train = config.get('train', {})

    # Calculate approximate parameters
    base = model.get('base', 32)
    dim = model.get('dim', 256)
    depth = model.get('depth', 2)

    # Rough estimate: params ≈ base² × dim × depth × constant
    # For BrainTumNet: empirical constant ≈ 15
    approx_params = (base * base * dim * depth * 15) // 1_000_000

    return {
        'name': name,
        'batch_size': train.get('batch_size', 'N/A'),
        'lr': train.get('lr', 'N/A'),
        'epochs': train.get('epochs', 'N/A'),
        'workers': train.get('workers', 'N/A'),
        'amp_dtype': train.get('amp_dtype', 'float16'),
        'model_type': model.get('model_type', 'v1'),
        'base': base,
        'dim': dim,
        'depth': depth,
        'n_heads': model.get('n_heads', 4),
        'approx_params': approx_params,
        'channels_last': train.get('channels_last', False),
        'cudnn_benchmark': train.get('cudnn_benchmark', False),
        'iou_weight': train.get('iou_weight', 1.0),
        'boundary_weight': train.get('boundary_weight', 0.0),
    }


def print_comparison_table():
    """Print comparison table of all configs."""

    configs_dir = Path('configs')

    # Define configs to compare
    config_files = {
        'Phase 1': 'phase1_iou_focus.yaml',
        'Phase 2 Small': 'phase2_small.yaml',
        'Phase 2 A100': 'phase2_a100_80gb.yaml',
    }

    # Load all configs
    configs = {}
    for name, filename in config_files.items():
        path = configs_dir / filename
        if path.exists():
            try:
                cfg = load_config(path)
                configs[name] = get_config_info(cfg, name)
            except Exception as e:
                print(f"⚠️  Failed to load {filename}: {e}")
        else:
            print(f"⚠️  Config not found: {filename}")

    if not configs:
        print("❌ No configs found!")
        return

    # Print header
    print("\n" + "="*120)
    print("📊 TRAINING CONFIGURATION COMPARISON")
    print("="*120 + "\n")

    # Training parameters
    print("🎮 TRAINING PARAMETERS")
    print("-" * 120)
    print(f"{'Parameter':<25} | {'Phase 1':>20} | {'Phase 2 Small':>20} | {'Phase 2 A100':>20}")
    print("-" * 120)

    params = [
        ('Batch Size', 'batch_size'),
        ('Learning Rate', 'lr'),
        ('Epochs', 'epochs'),
        ('Workers', 'workers'),
        ('Mixed Precision', 'amp_dtype'),
    ]

    for label, key in params:
        values = []
        for name in ['Phase 1', 'Phase 2 Small', 'Phase 2 A100']:
            if name in configs:
                val = configs[name].get(key, 'N/A')
                if isinstance(val, float):
                    val = f"{val:.2e}"
                values.append(str(val))
            else:
                values.append('N/A')

        print(f"{label:<25} | {values[0]:>20} | {values[1]:>20} | {values[2]:>20}")

    # Model architecture
    print("\n🏗️  MODEL ARCHITECTURE")
    print("-" * 120)
    print(f"{'Parameter':<25} | {'Phase 1':>20} | {'Phase 2 Small':>20} | {'Phase 2 A100':>20}")
    print("-" * 120)

    model_params = [
        ('Model Type', 'model_type'),
        ('Base Channels', 'base'),
        ('Transformer Dim', 'dim'),
        ('Depth', 'depth'),
        ('Attention Heads', 'n_heads'),
        ('Parameters (M)', 'approx_params'),
    ]

    for label, key in model_params:
        values = []
        for name in ['Phase 1', 'Phase 2 Small', 'Phase 2 A100']:
            if name in configs:
                val = configs[name].get(key, 'N/A')
                values.append(str(val))
            else:
                values.append('N/A')

        print(f"{label:<25} | {values[0]:>20} | {values[1]:>20} | {values[2]:>20}")

    # Hardware optimizations
    print("\n⚡ HARDWARE OPTIMIZATIONS")
    print("-" * 120)
    print(f"{'Feature':<25} | {'Phase 1':>20} | {'Phase 2 Small':>20} | {'Phase 2 A100':>20}")
    print("-" * 120)

    hw_params = [
        ('Channels Last', 'channels_last'),
        ('cuDNN Benchmark', 'cudnn_benchmark'),
    ]

    for label, key in hw_params:
        values = []
        for name in ['Phase 1', 'Phase 2 Small', 'Phase 2 A100']:
            if name in configs:
                val = configs[name].get(key, False)
                val_str = '✅' if val else '❌'
                values.append(val_str)
            else:
                values.append('N/A')

        print(f"{label:<25} | {values[0]:>20} | {values[1]:>20} | {values[2]:>20}")

    # Loss configuration
    print("\n🎯 LOSS CONFIGURATION")
    print("-" * 120)
    print(f"{'Component':<25} | {'Phase 1':>20} | {'Phase 2 Small':>20} | {'Phase 2 A100':>20}")
    print("-" * 120)

    loss_params = [
        ('IoU Weight', 'iou_weight'),
        ('Boundary Weight', 'boundary_weight'),
    ]

    for label, key in loss_params:
        values = []
        for name in ['Phase 1', 'Phase 2 Small', 'Phase 2 A100']:
            if name in configs:
                val = configs[name].get(key, 0.0)
                values.append(f"{val:.1f}")
            else:
                values.append('N/A')

        print(f"{label:<25} | {values[0]:>20} | {values[1]:>20} | {values[2]:>20}")

    # Expected performance
    print("\n📈 EXPECTED PERFORMANCE")
    print("-" * 120)
    print(f"{'Metric':<25} | {'Phase 1':>20} | {'Phase 2 Small':>20} | {'Phase 2 A100':>20}")
    print("-" * 120)

    performance = [
        ('Recommended GPU', 'RTX 3090', 'RTX 3090', 'A100 80GB'),
        ('GPU Memory (GB)', '~10-12', '~12-15', '~55-65'),
        ('Time per Epoch (h)', '~2-3', '~3-4', '~2-2.5'),
        ('Time per Fold (h)', '~40', '~48', '~18'),
        ('Total 5-Fold (days)', '~8', '~10', '~4'),
        ('Single Model IoU', '0.75-0.80', '0.80-0.82', '0.82-0.85'),
        ('With Ensemble IoU', '0.78-0.83', '0.83-0.85', '0.87-0.90'),
    ]

    for row in performance:
        label = row[0]
        v1, v2, v3 = row[1], row[2], row[3]
        print(f"{label:<25} | {v1:>20} | {v2:>20} | {v3:>20}")

    print("-" * 120)

    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("="*120)
    print("""
🎯 WHICH CONFIG TO USE?

Phase 1 (phase1_iou_focus.yaml):
  ✅ Use if: Just starting, want to test new loss functions
  ✅ GPU: RTX 3090, RTX 4090, or similar (24GB+)
  ✅ Time: ~8 days for 5-fold
  ✅ IoU: 0.75-0.80 (good baseline improvement)
  ⚠️  Note: Phase 2 includes all Phase 1 improvements

Phase 2 Small (phase2_small.yaml):
  ✅ Use if: Have RTX 3090/4090, want best results on consumer GPU
  ✅ GPU: RTX 3090, RTX 4090 (24GB+)
  ✅ Time: ~10 days for 5-fold
  ✅ IoU: 0.80-0.82 single, 0.83-0.85 ensemble
  ⭐ RECOMMENDED for most users with RTX 3090/4090

Phase 2 A100 (phase2_a100_80gb.yaml):
  ✅ Use if: Have access to A100 80GB GPU
  ✅ GPU: A100 80GB (MUST have 80GB, not 40GB)
  ✅ Time: ~4 days for 5-fold (2.5x faster!)
  ✅ IoU: 0.82-0.85 single, 0.87-0.90 ensemble ✅ TARGET!
  ⭐ BEST performance if you have A100
  💰 Cloud cost: ~$113 (Lambda Labs)

SUMMARY:
- RTX 3090/4090: Use Phase 2 Small
- A100 80GB: Use Phase 2 A100
- A100 40GB: Use Phase 2 Small (A100 config will OOM)
- Skip Phase 1: Phase 2 includes everything
""")

    print("="*120)


def main():
    print_comparison_table()


if __name__ == "__main__":
    main()
