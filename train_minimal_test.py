#!/usr/bin/env python
"""Minimal test of residual RL environment to verify setup."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import mujoco

def test_env_creation():
    """Test if we can create the SO101 environment."""
    print("Testing SO101 Residual Environment creation...")

    try:
        from lerobot.envs.so101_residual_env import SO101ResidualEnv
        print("✓ SO101ResidualEnv imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SO101ResidualEnv: {e}")
        return False

    try:
        # Create environment with minimal configuration
        env = SO101ResidualEnv(
            base_policy=None,  # Zero-action baseline
            alpha=1.0,  # Full residual
            act_scale=0.02,
            residual_penalty=0.001,
            randomize=False,  # No domain randomization for test
            render_mode=None,  # No rendering
        )
        print("✓ Environment created successfully")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False

    try:
        # Test reset
        obs, info = env.reset(seed=42)
        print(f"✓ Environment reset successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation range: [{obs.min():.2f}, {obs.max():.2f}]")
    except Exception as e:
        print(f"✗ Failed to reset environment: {e}")
        return False

    try:
        # Test step with zero action
        action = np.zeros(6)  # 6D action space
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Environment step successful")
        print(f"  Reward: {reward:.3f}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
    except Exception as e:
        print(f"✗ Failed to step environment: {e}")
        return False

    print("\n✅ All environment tests passed!")
    return True

def test_wandb_connection():
    """Test W&B connection."""
    print("\nTesting Weights & Biases connection...")

    # Check if API key is set
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        print("✗ WANDB_API_KEY not found in environment")
        print("  Loading from .env file...")
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("WANDB_API_KEY")
            if api_key:
                print("✓ WANDB_API_KEY loaded from .env")
            else:
                print("✗ WANDB_API_KEY not found in .env")
                return False
        except ImportError:
            print("✗ python-dotenv not installed, cannot load .env")
            return False
    else:
        print("✓ WANDB_API_KEY found in environment")

    try:
        import wandb
        print(f"✓ wandb module imported (version: {wandb.__version__})")

        # Test API connection
        wandb.login(key=api_key, verify=True)
        print("✓ Successfully authenticated with W&B")

        # Get user info
        api = wandb.Api()
        user = api.viewer
        print(f"✓ Logged in as: {user.get('username', 'unknown')}")

        return True
    except ImportError:
        print("✗ wandb not installed")
        return False
    except Exception as e:
        print(f"✗ Failed to connect to W&B: {e}")
        return False

def main():
    print("=" * 60)
    print("SO101 Residual RL Training Setup Test")
    print("=" * 60)

    # Test environment
    env_ok = test_env_creation()

    # Test W&B
    wandb_ok = test_wandb_connection()

    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Environment: {'✅ PASS' if env_ok else '❌ FAIL'}")
    print(f"  W&B Connection: {'✅ PASS' if wandb_ok else '❌ FAIL'}")

    if env_ok and wandb_ok:
        print("\n🎉 All systems ready for training!")
        print("\nYou can now run the full training with:")
        print("  python src/lerobot/scripts/train_so101_residual.py \\")
        print("    --base-policy zero --alpha 1.0 --total-timesteps 500000")
    else:
        print("\n⚠️  Some components need attention before training")

    print("=" * 60)

if __name__ == "__main__":
    main()