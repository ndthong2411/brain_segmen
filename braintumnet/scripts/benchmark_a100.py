"""
Benchmark script to diagnose A100 performance issues.
Run this on A100 server to identify bottlenecks.

Usage:
    python scripts/benchmark_a100.py --cfg configs/a100_max.yaml
"""

import argparse
import time
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from braintumnet.utils.io import load_yaml
from braintumnet.engine.trainer import build_model, build_dataloaders


def benchmark_dataloader(loader, name="Train", max_batches=10):
    """Benchmark data loading speed."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {name} DataLoader")
    print(f"{'='*60}")

    start = time.time()
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch_time = time.time() - start
        print(f"  Batch {i+1}/{max_batches}: {batch_time:.3f}s", flush=True)

        # Check batch properties
        if i == 0:
            img = batch["image"]
            msk = batch["mask"]
            print(f"  Batch shape: {img.shape}")
            print(f"  Batch dtype: {img.dtype}")
            print(f"  Mask shape: {msk.shape}")

        start = time.time()

    avg_time = (time.time() - start) / min(max_batches, len(loader))
    print(f"\nAverage batch load time: {avg_time:.3f}s")
    return avg_time


def benchmark_forward(model, device, batch_size=32, img_size=256, num_iters=10):
    """Benchmark model forward pass."""
    print(f"\n{'='*60}")
    print(f"Benchmarking Model Forward Pass")
    print(f"{'='*60}")

    model.eval()
    dummy_input = torch.randn(batch_size, 4, img_size, img_size).to(device)

    # Warmup
    print("Warming up GPU...", flush=True)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    # Benchmark
    print(f"Running {num_iters} iterations...", flush=True)
    times = []
    with torch.no_grad():
        for i in range(num_iters):
            start = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize()  # Wait for GPU to finish
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Iteration {i+1}/{num_iters}: {elapsed*1000:.1f}ms", flush=True)

    avg_time = sum(times) / len(times)
    print(f"\nAverage forward time: {avg_time*1000:.1f}ms ({1/avg_time:.1f} fps)")
    return avg_time


def benchmark_backward(model, device, batch_size=32, img_size=256, num_iters=10):
    """Benchmark model forward + backward pass."""
    print(f"\n{'='*60}")
    print(f"Benchmarking Model Forward + Backward Pass")
    print(f"{'='*60}")

    model.train()
    dummy_input = torch.randn(batch_size, 4, img_size, img_size).to(device)
    dummy_mask = torch.randint(0, 2, (batch_size, 1, img_size, img_size)).float().to(device)
    dummy_label = torch.randint(0, 2, (batch_size,)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Warmup
    print("Warming up GPU...", flush=True)
    for _ in range(5):
        optimizer.zero_grad()
        seg, cls = model(dummy_input)
        loss = seg.mean() + cls.mean()
        loss.backward()

    # Benchmark
    print(f"Running {num_iters} iterations...", flush=True)
    times = []
    for i in range(num_iters):
        start = time.time()
        optimizer.zero_grad()

        output = model(dummy_input)
        if len(output) == 3:  # Deep supervision
            seg, cls, _ = output
        else:
            seg, cls = output

        loss = seg.mean() + cls.mean()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}/{num_iters}: {elapsed*1000:.1f}ms", flush=True)

    avg_time = sum(times) / len(times)
    print(f"\nAverage forward+backward time: {avg_time*1000:.1f}ms ({1/avg_time:.1f} it/s)")
    return avg_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Config file")
    parser.add_argument("--fold", type=int, default=0, help="Fold number")
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"# A100 Performance Benchmark")
    print(f"# Config: {args.cfg}")
    print(f"{'#'*60}\n")

    # Load config
    cfg = load_yaml(args.cfg)
    cfg["data"]["fold"] = args.fold

    # Device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Available VRAM: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")
    else:
        print("WARNING: CUDA not available!")
        return

    # Build model
    print(f"\nBuilding model...", flush=True)
    model = build_model(cfg).to(device)

    # Check AMP
    amp_enabled = cfg["train"].get("amp", False)
    print(f"AMP (Mixed Precision): {'ENABLED ✓' if amp_enabled else 'DISABLED ✗'}")

    # Check channels_last
    channels_last = cfg["train"].get("use_channels_last", False)
    print(f"Channels Last: {'ENABLED ✓' if channels_last else 'DISABLED ✗'}")

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params/1e6:.1f}M")

    # Benchmark 1: DataLoader
    print(f"\n{'='*60}")
    print("BENCHMARK 1: Data Loading")
    print(f"{'='*60}")
    train_loader, val_loader = build_dataloaders(cfg, args.fold)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Batch size: {cfg['train']['batch_size']}")
    print(f"Workers: {cfg['train']['workers']}")

    data_time = benchmark_dataloader(train_loader, "Train", max_batches=10)

    # Benchmark 2: Forward pass
    forward_time = benchmark_forward(
        model, device,
        batch_size=cfg["train"]["batch_size"],
        img_size=cfg["data"]["img_size"],
        num_iters=10
    )

    # Benchmark 3: Forward + Backward pass
    backward_time = benchmark_backward(
        model, device,
        batch_size=cfg["train"]["batch_size"],
        img_size=cfg["data"]["img_size"],
        num_iters=10
    )

    # Summary
    print(f"\n{'#'*60}")
    print(f"# PERFORMANCE SUMMARY")
    print(f"{'#'*60}")
    print(f"Data loading:        {data_time*1000:6.1f}ms per batch")
    print(f"Forward pass:        {forward_time*1000:6.1f}ms per batch")
    print(f"Forward+Backward:    {backward_time*1000:6.1f}ms per batch")
    print(f"Total (estimated):   {(data_time + backward_time)*1000:6.1f}ms per batch")
    print(f"")
    print(f"Expected epoch time: {(data_time + backward_time) * len(train_loader) / 60:.1f} minutes")
    print(f"{'#'*60}\n")

    # Diagnosis
    print(f"{'='*60}")
    print("DIAGNOSIS")
    print(f"{'='*60}")

    if data_time > 0.5:
        print("⚠️  Data loading is SLOW (>500ms per batch)")
        print("   Solutions:")
        print("   - Increase workers in config")
        print("   - Check if data is on fast SSD (not HDD)")
        print("   - Check CPU cores available")
    else:
        print("✓ Data loading is OK")

    if forward_time > 0.1:
        print("⚠️  Forward pass is SLOW (>100ms per batch)")
        print("   Solutions:")
        print("   - Enable AMP (amp: true in config)")
        print("   - Enable channels_last (use_channels_last: true)")
        print("   - Reduce model size")
    else:
        print("✓ Forward pass is OK")

    if backward_time > 0.5:
        print("⚠️  Training step is SLOW (>500ms per batch)")
        print("   Solutions:")
        print("   - Enable AMP if not already")
        print("   - Check GPU utilization: nvidia-smi dmon -s u")
        print("   - Reduce batch size if memory swapping")
    else:
        print("✓ Training step is OK")

    expected_epoch_minutes = (data_time + backward_time) * len(train_loader) / 60
    if expected_epoch_minutes > 5:
        print(f"\n⚠️  Expected epoch time is HIGH: {expected_epoch_minutes:.1f} minutes")
        print("   Target for A100: < 2 minutes per epoch")
    else:
        print(f"\n✓ Expected epoch time is GOOD: {expected_epoch_minutes:.1f} minutes")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
