#!/usr/bin/env python
"""
Verify A100 Optimization Setup
Checks if all optimizations are properly configured
"""

import torch
import sys
import os

print("=" * 80)
print("A100 OPTIMIZATION VERIFICATION")
print("=" * 80)

# Check 1: PyTorch version (need 2.0+ for compile)
print("\n1. PyTorch Version Check")
pytorch_version = torch.__version__
print(f"   PyTorch version: {pytorch_version}")
major, minor = pytorch_version.split('.')[:2]
if int(major) >= 2:
    print("   ✅ PyTorch 2.0+ detected - torch.compile() available")
else:
    print("   ⚠️  PyTorch < 2.0 - torch.compile() not available (upgrade recommended)")

# Check 2: CUDA availability
print("\n2. CUDA Check")
if torch.cuda.is_available():
    print(f"   ✅ CUDA available")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    
    # Check if A100
    device_name = torch.cuda.get_device_name(0)
    if "A100" in device_name:
        print("   ✅ NVIDIA A100 detected!")
    else:
        print(f"   ⚠️  Not an A100 (found: {device_name})")
else:
    print("   ❌ CUDA not available")
    sys.exit(1)

# Check 3: BFloat16 support (A100 native)
print("\n3. BFloat16 Support (A100 Native)")
if hasattr(torch.cuda, 'is_bf16_supported'):
    if torch.cuda.is_bf16_supported():
        print("   ✅ BFloat16 supported (A100 optimization available)")
    else:
        print("   ⚠️  BFloat16 not supported (A100 specific feature)")
else:
    # Test by trying to create a bf16 tensor
    try:
        test_tensor = torch.zeros(1, dtype=torch.bfloat16, device='cuda')
        print("   ✅ BFloat16 supported (A100 optimization available)")
    except:
        print("   ⚠️  BFloat16 not supported")

# Check 4: TensorFloat32 (TF32) - A100 feature
print("\n4. TensorFloat32 (TF32) Status")
if hasattr(torch.backends.cuda, 'matmul'):
    tf32_enabled = torch.backends.cuda.matmul.allow_tf32
    print(f"   TF32 for matmul: {'✅ Enabled' if tf32_enabled else '⚠️  Disabled'}")
    if not tf32_enabled:
        print("   Set in config: cudnn_benchmark: true")
        
if hasattr(torch.backends.cudnn, 'allow_tf32'):
    tf32_cudnn = torch.backends.cudnn.allow_tf32
    print(f"   TF32 for cuDNN: {'✅ Enabled' if tf32_cudnn else '⚠️  Disabled'}")

# Check 5: Memory info
print("\n5. GPU Memory")
total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
print(f"   Total VRAM: {total_memory:.1f} GB")
if total_memory > 75:
    print("   ✅ 80GB A100 detected")
elif total_memory > 35:
    print("   ✅ 40GB A100 detected")
else:
    print(f"   ⚠️  Not an A100 ({total_memory:.1f} GB VRAM)")

allocated = torch.cuda.memory_allocated(0) / (1024**3)
reserved = torch.cuda.memory_reserved(0) / (1024**3)
print(f"   Allocated: {allocated:.2f} GB")
print(f"   Reserved: {reserved:.2f} GB")
print(f"   Available: {total_memory - reserved:.2f} GB")

# Check 6: Compute capability (A100 is 8.0)
print("\n6. Compute Capability")
capability = torch.cuda.get_device_capability(0)
capability_str = f"{capability[0]}.{capability[1]}"
print(f"   Compute capability: {capability_str}")
if capability[0] == 8 and capability[1] == 0:
    print("   ✅ A100 compute capability (8.0)")
elif capability[0] >= 7:
    print(f"   ✅ Modern GPU (cap {capability_str})")
else:
    print(f"   ⚠️  Older GPU (cap {capability_str})")

# Check 7: cuDNN
print("\n7. cuDNN")
if torch.backends.cudnn.is_available():
    print(f"   ✅ cuDNN available (version {torch.backends.cudnn.version()})")
    print(f"   Benchmark mode: {torch.backends.cudnn.benchmark}")
    print(f"   Deterministic mode: {torch.backends.cudnn.deterministic}")
else:
    print("   ❌ cuDNN not available")

# Check 8: Fused optimizer support
print("\n8. Fused Optimizer Support")
try:
    # Test if fused optimizer is available
    test_model = torch.nn.Linear(10, 10).cuda()
    test_opt = torch.optim.AdamW(test_model.parameters(), lr=1e-3, fused=True)
    print("   ✅ Fused AdamW available (A100 optimization)")
    del test_model, test_opt
except:
    print("   ⚠️  Fused optimizer not available (PyTorch too old)")

# Check 9: Data loading optimization
print("\n9. Data Loading")
import multiprocessing
cpu_count = multiprocessing.cpu_count()
print(f"   CPU cores: {cpu_count}")
print(f"   Recommended workers: {min(16, cpu_count)}")
print(f"   Recommended prefetch_factor: 4")

# Check 10: Test channels_last format
print("\n10. Channels Last Memory Format")
try:
    test_tensor = torch.randn(1, 4, 256, 256).cuda()
    test_tensor_cl = test_tensor.to(memory_format=torch.channels_last)
    print("   ✅ Channels last format supported")
    del test_tensor, test_tensor_cl
except:
    print("   ⚠️  Channels last format not supported")

# Check 11: Config file
print("\n11. Configuration File Check")
config_path = "configs/phase2_a100_o.yaml"
if os.path.exists(config_path):
    print(f"   ✅ Found: {config_path}")
    
    # Quick parse to check key settings
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    workers = cfg.get('train', {}).get('workers', 0)
    prefetch = cfg.get('train', {}).get('prefetch_factor', 2)
    batch_size = cfg.get('train', {}).get('batch_size', 8)
    amp_dtype = cfg.get('train', {}).get('amp_dtype', 'float16')
    use_compile = cfg.get('train', {}).get('use_compile', False)
    channels_last = cfg.get('train', {}).get('channels_last', False)
    
    print(f"\n   Config Settings:")
    print(f"   - workers: {workers} {'✅' if workers >= 12 else '⚠️ (recommend 16)'}")
    print(f"   - prefetch_factor: {prefetch} {'✅' if prefetch >= 4 else '⚠️ (recommend 4)'}")
    print(f"   - batch_size: {batch_size} {'✅' if batch_size >= 24 else '⚠️ (recommend 32)'}")
    print(f"   - amp_dtype: {amp_dtype} {'✅' if amp_dtype == 'bfloat16' else '⚠️ (recommend bfloat16)'}")
    print(f"   - use_compile: {use_compile} {'✅' if use_compile else '⚠️ (recommend true)'}")
    print(f"   - channels_last: {channels_last} {'✅' if channels_last else '⚠️ (recommend true)'}")
else:
    print(f"   ❌ Config file not found: {config_path}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Expected performance
print("\nExpected A100 Performance:")
print("  GPU Utilization: 90-100%")
print("  Power Usage: 300-350W")
print("  Memory Usage: 50-60GB (batch_size=32)")
print("  Training Speed: 3-4 batches/sec")
print("  Time/epoch: 1-1.5 hours")

print("\nTo monitor during training:")
print("  watch -n 1 nvidia-smi")

print("\nTo start training:")
print("  python scripts/train.py --cfg configs/phase2_a100_o.yaml --fold 0")

print("\n" + "=" * 80)
print("Verification complete!")
print("=" * 80)
