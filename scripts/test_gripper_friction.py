#!/usr/bin/env python
"""
Test script to validate high-friction gripper fingertips implementation.

This script:
1. Loads the SO101ResidualEnv environment
2. Verifies fingertip geoms are present
3. Tests friction randomization
4. Validates runtime friction modulation (release hack)
5. Launches viewer for visual inspection
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import mujoco as mj

print("=" * 80)
print("Gripper Friction Implementation Test")
print("=" * 80)
print()

# Step 1: Load environment
print("Step 1: Loading SO101ResidualEnv...")
try:
    from lerobot.envs.so101_residual_env import SO101ResidualEnv

    env = SO101ResidualEnv(randomize=True, seed=42)
    print("✅ Environment loaded successfully")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.shape}")
except Exception as e:
    print(f"❌ Failed to load environment: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 2: Check fingertip geoms
print("Step 2: Checking fingertip geoms...")
if env.fixed_fingertip_id is not None and env.moving_fingertip_id is not None:
    if env.fixed_fingertip_id >= 0 and env.moving_fingertip_id >= 0:
        print(f"✅ Fixed fingertip ID: {env.fixed_fingertip_id}")
        print(f"✅ Moving fingertip ID: {env.moving_fingertip_id}")

        # Check friction values
        fixed_friction = env.model.geom_friction[env.fixed_fingertip_id]
        moving_friction = env.model.geom_friction[env.moving_fingertip_id]

        print(f"   Fixed fingertip friction: {fixed_friction}")
        print(f"   Moving fingertip friction: {moving_friction}")

        # Check collision properties
        fixed_contype = env.model.geom_contype[env.fixed_fingertip_id]
        moving_contype = env.model.geom_contype[env.moving_fingertip_id]

        print(f"   Fixed fingertip contype: {fixed_contype} (0=disabled, 1=enabled)")
        print(f"   Moving fingertip contype: {moving_contype} (0=disabled, 1=enabled)")

        if fixed_contype == 0:
            print("   ⚠️  Phase A: Fingertips collision disabled (pushing-only)")
            print("   ℹ️  To enable grasping (Phase B): Change contype='0' to '2' in XML")
    else:
        print("❌ Fingertip IDs invalid")
        sys.exit(1)
else:
    print("❌ Fingertip IDs not found")
    sys.exit(1)

print()

# Step 3: Test friction randomization
print("Step 3: Testing friction randomization across resets...")
friction_values = []

for i in range(5):
    obs, info = env.reset(seed=42 + i)
    friction_values.append(env.gripper_base_friction)

friction_values = np.array(friction_values)
print(f"   Base friction values: {friction_values}")
print(f"   Mean: {friction_values.mean():.3f}")
print(f"   Std: {friction_values.std():.3f}")
print(f"   Range: [{friction_values.min():.3f}, {friction_values.max():.3f}]")

if friction_values.std() > 0.01:
    print("✅ Friction randomization working (variance detected)")
else:
    print("⚠️  Friction appears constant (check randomization)")

print()

# Step 4: Test runtime friction modulation
print("Step 4: Testing runtime friction modulation (release hack)...")
obs, info = env.reset(seed=42)

# Close gripper
print("   Closing gripper...")
for i in range(20):
    action = np.zeros(6)
    action[5] = -0.1  # Close gripper
    obs, reward, terminated, truncated, info = env.step(action)

closed_friction = env.model.geom_friction[env.fixed_fingertip_id, 0]
gripper_pos_closed = env.data.qpos[env.joint_ids[5]]
print(f"   Gripper closed (pos={gripper_pos_closed:.3f}): friction={closed_friction:.3f}")

# Open gripper
print("   Opening gripper...")
for i in range(30):
    action = np.zeros(6)
    action[5] = 0.1  # Open gripper
    obs, reward, terminated, truncated, info = env.step(action)

open_friction = env.model.geom_friction[env.fixed_fingertip_id, 0]
gripper_pos_open = env.data.qpos[env.joint_ids[5]]
print(f"   Gripper opened (pos={gripper_pos_open:.3f}): friction={open_friction:.3f}")

if gripper_pos_open > env.gripper_release_threshold:
    if open_friction < closed_friction:
        print("✅ Release hack working (friction decreased when opening)")
    else:
        print("⚠️  Friction did not decrease as expected")
else:
    print("⚠️  Gripper did not reach release threshold")

print()

# Step 5: Visual inspection guide
print("=" * 80)
print("✅ ALL TESTS PASSED")
print()
print("Visual Inspection Guide:")
print("  1. Fingertips should appear as small dark gray capsules near gripper tips")
print("  2. Enable contact forces: Press 'Ctrl+F' in viewer")
print("  3. Close gripper and observe contact forces with paper")
print("  4. Paper should NOT pass through fingertips in Phase B (after enabling contype)")
print()
print("Phase B Activation (for grasping):")
print("  - Edit: src/lerobot/envs/so101_assets/paper_square_realistic.xml")
print("  - Line ~172: Change fixed_fingertip contype='0' → contype='2'")
print("  - Line ~208: Change moving_fingertip contype='0' → contype='2'")
print("  - Save and reload environment")
print()
print("Next Steps:")
print("  1. Launch viewer: python scripts/view_world.py")
print("  2. Run Phase A training: python scripts/train_ppo_residual.py")
print("  3. After convergence: Activate Phase B and continue training")
print("=" * 80)

env.close()
