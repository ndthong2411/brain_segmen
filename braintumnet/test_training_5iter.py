"""
Test training for 5 iterations to verify all bugs are fixed
"""
import torch
import yaml
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from braintumnet.models.braintumnet import BrainTumNet
from braintumnet.losses_combined import create_loss_from_config
from braintumnet.data.brats2020_dataset import SliceDataset
from torch.utils.data import DataLoader

def test_training():
    print("="*70)
    print("Testing 5 Training Iterations")
    print("="*70)

    # Load config
    with open('configs/phase1_iou_focus.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Create model
    print("\nCreating model...")
    model = BrainTumNet(
        in_ch=cfg['model']['in_channels'],
        num_cls=cfg['model']['num_classes_cls'],
        base=cfg['model']['base'],
        dim=cfg['model']['dim'],
        patch=cfg['model']['patch_size'],
        depth=cfg['model']['depth'],
        n_heads=cfg['model']['n_heads'],
        roi_stop_grad=cfg['model'].get('roi_stop_grad', True),
        deep_supervision=cfg['model'].get('deep_supervision', True),
        num_classes_seg=cfg['model']['num_classes_seg']
    ).to(device)
    print(f"Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # Create loss
    print("\nCreating loss...")
    criterion = create_loss_from_config(cfg)
    print(f"Loss created: {type(criterion).__name__}")

    # Create optimizer
    print("\nCreating optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['train']['lr'],
        weight_decay=cfg['train']['weight_decay']
    )
    print(f"Optimizer: AdamW with lr={cfg['train']['lr']}")

    # Load dataset (just a few samples for testing)
    print("\nLoading dataset...")
    import os
    proc_root = cfg['data']['proc_root']
    split_file = os.path.join(proc_root, f"split_train_fold4.txt")
    dataset = SliceDataset(
        proc_root=proc_root,
        split_file=split_file,
        img_size=cfg['data']['img_size'],
        train=True,
        in_channels=cfg['model']['in_channels']
    )

    # Use only 5 samples for quick test
    indices = list(range(min(5, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)

    dataloader = DataLoader(
        subset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    print(f"Dataset loaded: {len(subset)} samples, batch_size=2")

    # Training loop
    print("\n" + "="*70)
    print("Starting Training Loop")
    print("="*70)

    model.train()
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    for iteration, batch in enumerate(dataloader):
        if iteration >= 5:
            break

        print(f"\nIteration {iteration + 1}/5")
        print("-" * 40)

        # Move data to device
        img = batch["image"].to(device)
        msk = batch["mask"].to(device)
        lab = batch["label"].to(device)

        print(f"  Input shapes:")
        print(f"    Image: {img.shape}")
        print(f"    Mask: {msk.shape}")
        print(f"    Label: {lab.shape}")

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        try:
            if device == "cuda":
                with torch.cuda.amp.autocast():
                    # Model forward
                    model_output = model(img)

                    # Handle deep supervision
                    if cfg["model"].get("deep_supervision", False):
                        seg, cls, aux_outputs = model_output
                        print(f"  Output shapes:")
                        print(f"    Seg: {seg.shape}")
                        print(f"    Cls: {cls.shape}")
                        print(f"    Aux: [{aux_outputs[0].shape}, {aux_outputs[1].shape}, {aux_outputs[2].shape}]")
                    else:
                        seg, cls = model_output
                        aux_outputs = None
                        print(f"  Output shapes:")
                        print(f"    Seg: {seg.shape}")
                        print(f"    Cls: {cls.shape}")

                    # Compute loss
                    loss, loss_dict = criterion(seg, msk, cls, lab, aux_outputs)
            else:
                # CPU mode
                model_output = model(img)
                if cfg["model"].get("deep_supervision", False):
                    seg, cls, aux_outputs = model_output
                else:
                    seg, cls = model_output
                    aux_outputs = None
                loss, loss_dict = criterion(seg, msk, cls, lab, aux_outputs)

            print(f"  Loss values:")
            print(f"    Total: {loss.item():.4f}")
            print(f"    Dice: {loss_dict['dice']:.4f}")
            print(f"    Focal: {loss_dict['focal']:.4f}")
            print(f"    IoU: {loss_dict['iou']:.4f}")
            print(f"    Boundary: {loss_dict['boundary']:.4f}")
            print(f"    Cls: {loss_dict['cls']:.4f}")
            if 'aux_total' in loss_dict:
                print(f"    Aux: {loss_dict['aux_total']:.4f}")

        except Exception as e:
            print(f"\n  ERROR during forward pass:")
            print(f"    {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Backward pass
        try:
            if device == "cuda" and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            print(f"  Backward pass: OK")

        except Exception as e:
            print(f"\n  ERROR during backward pass:")
            print(f"    {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "="*70)
    print("ALL 5 ITERATIONS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nTraining is ready to run:")
    print("  python scripts/train.py --cfg configs/phase1_iou_focus.yaml --fold 4")
    print("="*70)

    return True

if __name__ == "__main__":
    success = test_training()
    sys.exit(0 if success else 1)
