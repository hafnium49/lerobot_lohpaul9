#!/usr/bin/env python
"""
Run SO101 Residual RL Training with GPU support
This script ensures GPU is properly initialized for WSL2
"""

import sys
import os
from pathlib import Path

# Set up environment for WSL2 GPU support
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test GPU availability before starting
import torch
print("=" * 60)
print("GPU Setup Check")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA compiled version: {torch.version.cuda}")

# Force CUDA initialization
if torch.cuda.is_available():
    print(f"✅ CUDA available")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    device = torch.device("cuda")

    # Test GPU with a simple operation
    test_tensor = torch.randn(10, 10).to(device)
    result = test_tensor @ test_tensor.T
    print(f"✅ GPU computation test passed")
else:
    print("⚠️  WARNING: CUDA not available, will use CPU")
    print("For WSL2, make sure:")
    print("  1. Windows has NVIDIA GPU drivers installed")
    print("  2. WSL2 is updated (wsl --update)")
    print("  3. nvidia-smi works in Windows")
    device = torch.device("cpu")

print("=" * 60)
print()

# Now run the actual training
if __name__ == "__main__":
    # Import training components
    from lerobot.scripts.train_so101_residual import main

    # Set up arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-policy", type=str, default="zero")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="runs/zero_policy_gpu_test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--act-scale", type=float, default=0.02)
    parser.add_argument("--residual-penalty", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--checkpoint-freq", type=int, default=25000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--randomize", action="store_true", default=True)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    print("Training Configuration:")
    print(f"  Base Policy: {args.base_policy}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Total Timesteps: {args.total_timesteps:,}")
    print(f"  Parallel Environments: {args.n_envs}")
    print(f"  Device: {device}")
    print()

    # Run training
    main(args)