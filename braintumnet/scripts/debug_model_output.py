"""
Debug script to test model output and predictions
"""
import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import numpy as np
from braintumnet.utils.io import load_yaml
from braintumnet.models import build_model

def test_model_output():
    """Test model output shapes and predictions"""

    # Load config
    config_path = ROOT / "configs" / "models" / "segunetv2_phase2.yaml"
    base_config_path = ROOT / "configs" / "base.yaml"

    print("="*70)
    print("Model Output Debug Test")
    print("="*70)

    # Load configs
    base_cfg = load_yaml(str(base_config_path))
    model_cfg = load_yaml(str(config_path))

    # Merge configs
    cfg = {**base_cfg, **model_cfg}
    cfg['model']['model_type'] = 'v2'

    print(f"Model type: {cfg['model']['model_type']}")
    print(f"Model config:")
    print(f"  in_channels: {cfg['model']['in_channels']}")
    print(f"  num_classes_seg: {cfg['model']['num_classes_seg']}")
    print(f"  base: {cfg['model']['base']}")
    print(f"  dim: {cfg['model']['dim']}")
    print(f"  depth: {cfg['model']['depth']}")
    print(f"  n_heads: {cfg['model']['n_heads']}")
    print(f"  boundary_refinement: {cfg['model'].get('boundary_refinement', False)}")
    print(f"  use_multiscale_transformer: {cfg['model'].get('use_multiscale_transformer', False)}")
    print(f"  use_attention_gates: {cfg['model'].get('use_attention_gates', False)}")

    # Build model
    print("\nBuilding model...")
    try:
        model = build_model(cfg)
        print("✓ Model built successfully")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params/1e6:.1f}M")
    except Exception as e:
        print(f"✗ Failed to build model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test forward pass
    print("\n" + "="*70)
    print("Testing forward pass...")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = model.to(device)
    model.eval()

    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 4, 256, 256).to(device)
    print(f"\nInput shape: {x.shape}")

    with torch.no_grad():
        try:
            output = model(x)

            if isinstance(output, tuple):
                if len(output) == 3:
                    seg, cls, aux = output
                    print(f"\n✓ Model output (with deep supervision):")
                    print(f"  Segmentation: {seg.shape}")
                    print(f"  Classification: {cls.shape}")
                    print(f"  Auxiliary outputs: {len(aux)} outputs")
                    for i, aux_out in enumerate(aux):
                        print(f"    Aux {i}: {aux_out.shape}")

                    # Check segmentation output
                    print(f"\n  Segmentation logits statistics:")
                    print(f"    Mean: {seg.mean().item():.4f}")
                    print(f"    Std: {seg.std().item():.4f}")
                    print(f"    Min: {seg.min().item():.4f}")
                    print(f"    Max: {seg.max().item():.4f}")

                    # Apply softmax and check predictions
                    seg_prob = torch.softmax(seg, dim=1)
                    seg_pred = torch.argmax(seg_prob, dim=1)  # (B, H, W)

                    print(f"\n  Predictions (after argmax):")
                    for b in range(batch_size):
                        pred_np = seg_pred[b].cpu().numpy()
                        unique, counts = np.unique(pred_np, return_counts=True)
                        print(f"    Sample {b}:")
                        for u, c in zip(unique, counts):
                            pct = (c / pred_np.size) * 100
                            print(f"      Class {u}: {c:6d} pixels ({pct:5.2f}%)")

                elif len(output) == 2:
                    seg, cls = output
                    print(f"\n✓ Model output (no deep supervision):")
                    print(f"  Segmentation: {seg.shape}")
                    print(f"  Classification: {cls.shape}")
            else:
                print(f"✗ Unexpected output format: {type(output)}")

        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return

    print("\n" + "="*70)
    print("✓ Model output test complete!")
    print("="*70)

if __name__ == "__main__":
    test_model_output()
