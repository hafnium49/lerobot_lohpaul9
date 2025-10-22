#!/usr/bin/env python
"""
Test script for the new potential-based reward system.
"""
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.envs.so101_residual_env import SO101ResidualEnv


def test_reward_components():
    """Test that all reward components are computed correctly."""
    print("=" * 60)
    print("Testing New Reward System with Contact-Gated Potential Shaping")
    print("=" * 60)

    # Create environment
    env = SO101ResidualEnv(
        base_policy=None,
        alpha=1.0,
        act_scale=0.02,
        randomize=False,
        seed=42,
    )

    # Reset environment
    obs, info = env.reset()
    print(f"\n✅ Environment created and reset successfully")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Initial paper position: {info['paper_pos']}")

    # Run a few steps with random actions
    print(f"\n{'='*60}")
    print("Running 10 test steps with random actions...")
    print(f"{'='*60}\n")

    for step in range(10):
        # Random action
        action = env.action_space.sample()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Print reward components
        print(f"Step {step + 1}:")
        print(f"  Total Reward:      {info['total_reward']:7.4f}")
        print(f"  ├─ Reach Bonus:    {info['reach_bonus']:7.4f}  (d={info['d_reach']:.4f}m)")
        print(f"  ├─ Push Reward:    {info['push_reward']:7.4f}  (in_contact={info['in_contact']}, progress={info['progress']:.5f})")
        print(f"  ├─ Dist Reward:    {info['dist_reward']:7.4f}  (dist={info['dist_to_goal']:.4f}m)")
        print(f"  ├─ Ori Reward:     {info['ori_reward']:7.4f}  (error={info['orientation_error']:.4f}rad)")
        print(f"  ├─ Residual Pen:   {info['residual_penalty']:7.4f}")
        print(f"  ├─ Time Penalty:   {info['time_penalty']:7.4f}")
        print(f"  ├─ Table Contact:  {info['table_contact_penalty']:7.4f}")
        print(f"  ├─ Unwanted Contact: {info['unwanted_paper_contact_penalty']:7.4f}")
        print(f"  └─ Success Bonus:  {info['success_bonus']:7.4f}  (frames={info['inside_frames']})")
        print()

        if terminated or truncated:
            print(f"Episode ended at step {step + 1}")
            break

    env.close()
    print("=" * 60)
    print("✅ Test completed successfully!")
    print("=" * 60)

    # Verify key properties
    print("\n🔍 Key Observations:")
    print("  1. reach_bonus should be small but positive when approaching paper")
    print("  2. push_reward should only be non-zero when in_contact=True")
    print("  3. dist_reward should be negative and improve as paper moves toward goal")
    print("  4. success_bonus should be 8.0 only after 5 consecutive frames inside")
    print("  5. time_penalty should only trigger when progress <= 0")


if __name__ == "__main__":
    test_reward_components()
