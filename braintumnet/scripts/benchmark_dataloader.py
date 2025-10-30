"""
Benchmark DataLoader Performance
==================================

Compare different data loading backends:
- PNG: Standard PNG files
- PNG + Cache: PNG with LRU cache
- LMDB: Lightning Memory-Mapped Database

Usage:
    # Benchmark PNG backend
    python scripts/benchmark_dataloader.py \
        --backend png \
        --data_dir braintumnet/data/processed_multiclass_with_grades \
        --batch_size 16 \
        --num_workers 8

    # Benchmark LMDB backend
    python scripts/benchmark_dataloader.py \
        --backend lmdb \
        --data_dir braintumnet/data/lmdb_multiclass_with_grades \
        --batch_size 16 \
        --num_workers 16

Author: BrainTumNet Optimization
Date: 2025-10-30
"""

import os
import sys
import time
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from braintumnet.data.brats2020_dataset import SliceDataset
from braintumnet.data.lmdb_dataset import LMDBDataset


def benchmark_dataloader(
    dataset,
    batch_size=16,
    num_workers=8,
    num_batches=100,
    prefetch_factor=4
):
    """Benchmark DataLoader performance.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        num_workers: Number of workers
        num_batches: Number of batches to benchmark
        prefetch_factor: Prefetch factor

    Returns:
        stats: Dict with benchmark results
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

    print(f"\nBenchmarking DataLoader...")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Prefetch factor: {prefetch_factor}")
    print(f"  Batches to test: {num_batches}")

    # Warmup (not counted)
    print(f"\nWarming up...")
    for i, batch in enumerate(loader):
        if i >= 10:
            break

    # Benchmark
    print(f"\nBenchmarking...")
    batch_times = []
    total_samples = 0

    start_time = time.time()

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break

        batch_start = time.time()

        # Simulate minimal processing (just access data)
        _ = batch['image'].shape
        _ = batch['mask'].shape

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        total_samples += batch['image'].size(0)

    total_time = time.time() - start_time

    # Calculate statistics
    batch_times = np.array(batch_times)
    samples_per_sec = total_samples / total_time

    stats = {
        'total_time': total_time,
        'total_samples': total_samples,
        'num_batches': len(batch_times),
        'samples_per_sec': samples_per_sec,
        'batch_time_mean': batch_times.mean() * 1000,  # ms
        'batch_time_std': batch_times.std() * 1000,    # ms
        'batch_time_min': batch_times.min() * 1000,    # ms
        'batch_time_max': batch_times.max() * 1000,    # ms
    }

    return stats


def print_stats(backend_name, stats):
    """Print benchmark statistics.

    Args:
        backend_name: Name of backend (e.g., "PNG", "LMDB")
        stats: Stats dict from benchmark_dataloader
    """
    print(f"\n{'='*60}")
    print(f"  {backend_name} Backend Results")
    print(f"{'='*60}")
    print(f"Total time:        {stats['total_time']:.2f} s")
    print(f"Total samples:     {stats['total_samples']}")
    print(f"Num batches:       {stats['num_batches']}")
    print(f"Throughput:        {stats['samples_per_sec']:.1f} samples/s")
    print(f"\nBatch time (ms):")
    print(f"  Mean:  {stats['batch_time_mean']:.2f} ± {stats['batch_time_std']:.2f}")
    print(f"  Min:   {stats['batch_time_min']:.2f}")
    print(f"  Max:   {stats['batch_time_max']:.2f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark DataLoader performance")
    parser.add_argument("--backend", type=str, required=True,
                       choices=["png", "lmdb"],
                       help="Backend to benchmark")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Data directory")
    parser.add_argument("--fold", type=int, default=0,
                       help="Fold number")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of workers")
    parser.add_argument("--num_batches", type=int, default=100,
                       help="Number of batches to benchmark")
    parser.add_argument("--prefetch_factor", type=int, default=4,
                       help="Prefetch factor")
    parser.add_argument("--cache_size", type=int, default=1000,
                       help="Cache size for PNG backend")

    args = parser.parse_args()

    # Prepare split file
    split_file = os.path.join(args.data_dir, f"train_fold{args.fold}.csv")
    if not os.path.exists(split_file):
        print(f"Error: Split file not found: {split_file}")
        sys.exit(1)

    # Create dataset
    print(f"\nCreating {args.backend.upper()} dataset...")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Split file: {split_file}")

    if args.backend == "png":
        dataset = SliceDataset(
            proc_root=args.data_dir,
            split_file=split_file,
            img_size=256,
            rotate_deg=0,
            hflip_p=0.0,
            vflip_p=0.0,
            train=False,  # No augmentation for fair comparison
            in_channels=4,
            cache_size=args.cache_size
        )
        backend_name = f"PNG (cache={args.cache_size})"

    elif args.backend == "lmdb":
        dataset = LMDBDataset(
            lmdb_root=args.data_dir,
            split_file=split_file,
            img_size=256,
            rotate_deg=0,
            hflip_p=0.0,
            vflip_p=0.0,
            train=False,  # No augmentation for fair comparison
            in_channels=4
        )
        backend_name = "LMDB"

    print(f"  Dataset size: {len(dataset)}")

    # Run benchmark
    stats = benchmark_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_batches=args.num_batches,
        prefetch_factor=args.prefetch_factor
    )

    # Print results
    print_stats(backend_name, stats)

    # Save results to file
    import json
    results_file = f"benchmark_{args.backend}_b{args.batch_size}_w{args.num_workers}.json"
    with open(results_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
