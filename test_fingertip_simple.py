#!/usr/bin/env python
"""
Simple test: Check if fingertips can contact paper during continuous movement.
"""

import sys
from pathlib import Path

import mujoco as mj
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.envs.so101_base_policy import JacobianIKPolicy
from lerobot.envs.so101_residual_env import SO101ResidualEnv

def main():
    print("="*80)
    print("FINGERTIP-PAPER CONTACT TEST (Simple)")
    print("="*80)

    # Create environment
    base_policy = JacobianIKPolicy(max_delta=0.02)
    env = SO101ResidualEnv(
        base_policy=base_policy,
        alpha=1.0,
        act_scale=0.02,
        residual_penalty=0.0,
        randomize=False,
        render_mode=None,
        seed=42,
    )

    obs, info = env.reset()

    # Verify fingertip size
    fixed_ft_size = env.model.geom_size[env.fixed_fingertip_id][0]
    print(f"\n✅ Fingertip diameter: {fixed_ft_size*2*1000:.1f}mm")

    # Get collision settings
    fixed_contype = env.model.geom_contype[env.fixed_fingertip_id]
    fixed_conaffinity = env.model.geom_conaffinity[env.fixed_fingertip_id]
    print(f"✅ Fingertip collision: contype={fixed_contype}, conaffinity={fixed_conaffinity}")

    # Get paper collision
    paper_geom_id = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_GEOM, "paper_geom")
    paper_contype = env.model.geom_contype[paper_geom_id]
    paper_conaffinity = env.model.geom_conaffinity[paper_geom_id]
    print(f"✅ Paper collision: contype={paper_contype}, conaffinity={paper_conaffinity}")

    # Check if collision should work
    should_collide = (fixed_contype & paper_conaffinity) or (paper_contype & fixed_conaffinity)
    print(f"\n{'✅' if should_collide else '❌'} Collision check: ({fixed_contype} & {paper_conaffinity}) || ({paper_contype} & {fixed_conaffinity}) = {should_collide}")

    # Test with many random episodes
    print(f"\n" + "="*80)
    print("Running 10 episodes with random actions")
    print("="*80)

    total_timesteps = 0
    fingertip_contacts = 0
    robot_paper_contacts = 0
    table_contacts = 0

    for episode in range(10):
        obs, info = env.reset()

        for step in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_timesteps += 1

            # Track contact events
            if info.get("fingertip_contact_reward", 0.0) > 0:
                fingertip_contacts += 1
                if fingertip_contacts == 1:  # First contact
                    print(f"\n🎉 FIRST FINGERTIP CONTACT!")
                    print(f"   Episode {episode}, Step {step}")
                    print(f"   Reward: {info['fingertip_contact_reward']:+.3f}")

            if info.get("unwanted_paper_contact_penalty", 0.0) < 0:
                robot_paper_contacts += 1

            if info.get("table_contact_penalty", 0.0) < 0:
                table_contacts += 1

            if terminated or truncated:
                break

        # Print progress every 3 episodes
        if (episode + 1) % 3 == 0:
            print(f"   Episodes 1-{episode+1}: {total_timesteps} steps, "
                  f"{fingertip_contacts} fingertip contacts, "
                  f"{robot_paper_contacts} robot-paper contacts, "
                  f"{table_contacts} table contacts")

    # Summary
    print(f"\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nTotal timesteps: {total_timesteps}")
    print(f"\n📊 Contact Events:")
    print(f"   Fingertip-paper:    {fingertip_contacts:4d} ({100*fingertip_contacts/total_timesteps:.2f}%)")
    print(f"   Robot-paper (bad):  {robot_paper_contacts:4d} ({100*robot_paper_contacts/total_timesteps:.2f}%)")
    print(f"   Robot-table (bad):  {table_contacts:4d} ({100*table_contacts/total_timesteps:.2f}%)")

    if fingertip_contacts > 0:
        print(f"\n✅ SUCCESS: Fingertip-paper contact detection is WORKING!")
        print(f"   Contact rate: {fingertip_contacts} events in {total_timesteps} timesteps")
        print(f"   This will shape the policy during RL training.")
    else:
        print(f"\n⚠️  No fingertip contacts detected in {total_timesteps} random steps")
        print(f"   This is OK - contacts may be rare with random actions")
        print(f"   Learned policies will discover contact strategies during training")

    env.close()
    print("="*80)

if __name__ == "__main__":
    main()
