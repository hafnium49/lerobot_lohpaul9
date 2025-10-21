#!/usr/bin/env python
"""
Test script to verify contact detection and contact-based rewards are working.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.envs.so101_base_policy import JacobianIKPolicy
from lerobot.envs.so101_residual_env import SO101ResidualEnv

def test_contact_detection():
    """Test contact detection functionality."""
    print("="*80)
    print("CONTACT DETECTION TEST")
    print("="*80)

    # Create environment
    base_policy = JacobianIKPolicy(max_delta=0.02)
    env = SO101ResidualEnv(
        base_policy=base_policy,
        alpha=1.0,  # Pure RL (no base policy)
        act_scale=0.02,
        residual_penalty=0.0,
        randomize=False,  # Disable randomization for consistent testing
        render_mode=None,
        seed=42,
    )

    # Reset environment
    obs, info = env.reset()
    print(f"\n✅ Environment created and reset successfully")
    print(f"   Robot collision geoms tracked: {len(env.robot_collision_geom_ids)}")
    print(f"   Table geom ID: {env.table_geom_id}")
    print(f"   Fixed fingertip ID: {env.fixed_fingertip_id}")
    print(f"   Moving fingertip ID: {env.moving_fingertip_id}")

    # Test 1: Run 100 random steps and check for contacts
    print("\n" + "="*80)
    print("TEST 1: Random Actions (100 steps)")
    print("="*80)

    contact_summary = {
        "robot_table": 0,
        "robot_paper": 0,
        "fingertip_paper": 0,
    }
    reward_summary = {
        "table_penalty_sum": 0.0,
        "unwanted_paper_penalty_sum": 0.0,
        "fingertip_reward_sum": 0.0,
    }

    for step in range(100):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Track contact events
        if info.get("table_contact_penalty", 0.0) < 0:
            contact_summary["robot_table"] += 1
            reward_summary["table_penalty_sum"] += info["table_contact_penalty"]

        if info.get("unwanted_paper_contact_penalty", 0.0) < 0:
            contact_summary["robot_paper"] += 1
            reward_summary["unwanted_paper_penalty_sum"] += info["unwanted_paper_contact_penalty"]

        if info.get("fingertip_contact_reward", 0.0) > 0:
            contact_summary["fingertip_paper"] += 1
            reward_summary["fingertip_reward_sum"] += info["fingertip_contact_reward"]

        if terminated or truncated:
            break

    print(f"\n📊 Contact Events:")
    print(f"   Robot-table contacts:     {contact_summary['robot_table']:3d} timesteps")
    print(f"   Robot-paper contacts:     {contact_summary['robot_paper']:3d} timesteps")
    print(f"   Fingertip-paper contacts: {contact_summary['fingertip_paper']:3d} timesteps")

    print(f"\n💰 Reward Impact:")
    print(f"   Table contact penalty:    {reward_summary['table_penalty_sum']:+.3f}")
    print(f"   Unwanted paper penalty:   {reward_summary['unwanted_paper_penalty_sum']:+.3f}")
    print(f"   Fingertip contact reward: {reward_summary['fingertip_reward_sum']:+.3f}")

    # Test 2: Force arm down to trigger table contact
    print("\n" + "="*80)
    print("TEST 2: Force Arm Down (should trigger table contact)")
    print("="*80)

    env.reset()
    # Apply large downward action on joint 1 (shoulder lift)
    for i in range(20):
        action = np.zeros(6)
        action[1] = -1.0  # Maximum downward action on shoulder lift
        obs, reward, terminated, truncated, info = env.step(action)

        if info.get("table_contact_penalty", 0.0) < 0:
            print(f"   Step {i:2d}: ✅ Table contact detected! Penalty = {info['table_contact_penalty']:.3f}")
            break
    else:
        print(f"   ⚠️  No table contact detected in 20 steps")

    # Test 3: Check fingertip contact with paper
    print("\n" + "="*80)
    print("TEST 3: Move Gripper Toward Paper (should trigger fingertip contact)")
    print("="*80)

    env.reset()
    # Get paper position
    paper_pos = env.data.xpos[env.paper_body_id]
    print(f"   Paper position: ({paper_pos[0]:.3f}, {paper_pos[1]:.3f}, {paper_pos[2]:.3f})")

    # Try to move toward paper
    for i in range(50):
        # Simple approach: move forward and down
        action = np.zeros(6)
        action[0] = 0.3  # Pan toward paper
        action[1] = -0.2  # Move down
        action[2] = -0.3  # Extend elbow
        obs, reward, terminated, truncated, info = env.step(action)

        if info.get("fingertip_contact_reward", 0.0) > 0:
            print(f"   Step {i:2d}: ✅ Fingertip contact detected! Reward = {info['fingertip_contact_reward']:.3f}")
            break
    else:
        print(f"   ⚠️  No fingertip contact detected in 50 steps")

    env.close()

    print("\n" + "="*80)
    print("CONTACT DETECTION TEST COMPLETE")
    print("="*80)
    print("\nℹ️  Notes:")
    print("   - If all contact types are detected, the system is working!")
    print("   - If no contacts detected, check collision settings in XML")
    print("   - Fingertip contact may be rare with random actions")
    print("="*80)

if __name__ == "__main__":
    test_contact_detection()
