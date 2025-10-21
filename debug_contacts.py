#!/usr/bin/env python
"""
Debug script to check contact detection in detail.
"""

import sys
from pathlib import Path

import numpy as np
import mujoco as mj

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.envs.so101_base_policy import JacobianIKPolicy
from lerobot.envs.so101_residual_env import SO101ResidualEnv

def debug_contacts():
    """Debug contact detection."""
    print("="*80)
    print("CONTACT DEBUGGING")
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

    print(f"\n📋 Geom IDs:")
    print(f"   Robot collision geoms: {env.robot_collision_geom_ids}")
    print(f"   Table geom: {env.table_geom_id}")
    print(f"   Paper body: {env.paper_body_id}")
    print(f"   Fixed fingertip: {env.fixed_fingertip_id}")
    print(f"   Moving fingertip: {env.moving_fingertip_id}")

    # Force arm down aggressively
    print(f"\n🔽 Forcing arm down...")
    for i in range(50):
        action = np.zeros(6)
        action[1] = -1.0  # Maximum downward on shoulder lift
        action[2] = -1.0  # Maximum downward on elbow
        obs, reward, terminated, truncated, info = env.step(action)

        # Check raw contact buffer
        if i % 10 == 0:
            print(f"\n   Step {i}:")
            print(f"      Number of contacts: {env.data.ncon}")

            if env.data.ncon > 0:
                print(f"      Active contacts:")
                for j in range(min(5, env.data.ncon)):  # Show first 5
                    contact = env.data.contact[j]
                    geom1 = contact.geom1
                    geom2 = contact.geom2

                    # Get geom names
                    name1 = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_GEOM, geom1) or f"geom_{geom1}"
                    name2 = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_GEOM, geom2) or f"geom_{geom2}"

                    print(f"         [{j}] {name1} ({geom1}) <-> {name2} ({geom2})")

                    # Check if this is a robot-table contact
                    is_robot_table = ((geom1 in env.robot_collision_geom_ids and geom2 == env.table_geom_id) or
                                      (geom2 in env.robot_collision_geom_ids and geom1 == env.table_geom_id))
                    if is_robot_table:
                        print(f"            ✅ This is a robot-table contact!")

            # Check joint positions
            shoulder_lift = env.data.qpos[env.joint_ids[1]]
            elbow = env.data.qpos[env.joint_ids[2]]
            print(f"      Joint positions: shoulder_lift={shoulder_lift:.3f}, elbow={elbow:.3f}")

            # Check body heights
            for body_name in ["lower_arm", "wrist", "gripper"]:
                try:
                    body_id = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_BODY, body_name)
                    z_pos = env.data.xpos[body_id][2]
                    print(f"      {body_name:12s} Z = {z_pos:+.4f}m")
                except:
                    pass

    env.close()

if __name__ == "__main__":
    debug_contacts()
