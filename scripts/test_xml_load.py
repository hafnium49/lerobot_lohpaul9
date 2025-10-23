#!/usr/bin/env python3
"""Test that the XML loads without errors."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.envs.so101_residual_env import SO101ResidualEnv

def main():
    print("Testing XML loading...")

    try:
        # Create environment
        env = SO101ResidualEnv(render_mode="rgb_array")
        print("✅ XML loaded successfully!")

        # Reset environment to verify everything works
        obs, info = env.reset()
        print("✅ Environment reset successfully!")

        # Check observation structure
        print(f"\nObservation shape: {obs.shape}, dtype: {obs.dtype}")

        # Close environment
        env.close()
        print("\n✅ All tests passed!")
        return 0

    except Exception as e:
        print(f"\n❌ Error loading XML: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
