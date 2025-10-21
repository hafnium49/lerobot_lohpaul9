#!/usr/bin/env python
"""
Directly manipulate physics to force contacts and verify reward system.
"""

import sys
from pathlib import Path

import mujoco as mj
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.envs.so101_base_policy import JacobianIKPolicy
from lerobot.envs.so101_residual_env import SO101ResidualEnv

def force_contact_test():
    """Force contacts by directly manipulating MuJoCo state."""
    print("="*80)
    print("DIRECT CONTACT MANIPULATION TEST")
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

    print("\n📋 Environment Setup:")
    print(f"   Robot collision geoms: {env.robot_collision_geom_ids}")
    print(f"   Table geom: {env.table_geom_id}")
    print(f"   Fixed fingertip geom: {env.fixed_fingertip_id}")
    print(f"   Moving fingertip geom: {env.moving_fingertip_id}")
    print(f"   Paper body: {env.paper_body_id}")

    # Check fingertip collision settings
    print("\n🔍 Fingertip Collision Settings:")
    for name, geom_id in [("fixed_fingertip", env.fixed_fingertip_id),
                           ("moving_fingertip", env.moving_fingertip_id)]:
        if geom_id is not None and geom_id >= 0:
            contype = env.model.geom_contype[geom_id]
            conaffinity = env.model.geom_conaffinity[geom_id]
            size = env.model.geom_size[geom_id]
            print(f"   {name}: contype={contype}, conaffinity={conaffinity}, size={size[:1]}")

    # Check paper collision settings
    print("\n🔍 Paper Collision Settings:")
    for geom_id in range(env.model.ngeom):
        body_id = env.model.geom_bodyid[geom_id]
        if body_id == env.paper_body_id:
            geom_name = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
            contype = env.model.geom_contype[geom_id]
            conaffinity = env.model.geom_conaffinity[geom_id]
            print(f"   {geom_name}: contype={contype}, conaffinity={conaffinity}")

    # =========================================================================
    # TEST 1: Move paper DIRECTLY onto fingertip
    # =========================================================================
    print("\n" + "="*80)
    print("TEST 1: Move Paper Onto Fingertip (direct manipulation)")
    print("="*80)

    env.reset()

    # Get fingertip position (use fixed fingertip)
    fingertip_id = env.fixed_fingertip_id
    if fingertip_id >= 0:
        # Fingertip position is in body frame, need to get world position
        # For now, let's get the gripper EE position as proxy
        ee_pos = env.data.site_xpos[env.ee_site_id].copy()
        print(f"\n📍 End-effector at: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")

        # Move paper to EE position (slightly below to ensure contact)
        paper_qpos_start = 7  # Paper starts at qpos index 7 (after 6 robot joints)

        # Set paper position
        env.data.qpos[paper_qpos_start] = ee_pos[0]  # X
        env.data.qpos[paper_qpos_start + 1] = ee_pos[1]  # Y
        env.data.qpos[paper_qpos_start + 2] = ee_pos[2] - 0.10  # Z (10cm below EE)

        # Forward kinematics
        mj.mj_forward(env.model, env.data)

        print(f"   Moved paper to: ({env.data.qpos[paper_qpos_start]:.3f}, "
              f"{env.data.qpos[paper_qpos_start+1]:.3f}, {env.data.qpos[paper_qpos_start+2]:.3f})")

        # Step simulation to let contacts form
        print(f"\n   Stepping simulation...")
        for i in range(10):
            action = np.zeros(6)
            obs, reward, terminated, truncated, info = env.step(action)

            if i == 0:
                print(f"\n   📊 After 1 step:")
                print(f"      Contacts detected: {env.data.ncon}")
                print(f"      Fingertip reward: {info.get('fingertip_contact_reward', 0.0):+.3f}")
                print(f"      Robot-paper penalty: {info.get('unwanted_paper_contact_penalty', 0.0):+.3f}")
                print(f"      Total reward: {info.get('total_reward', 0.0):+.3f}")

                if env.data.ncon > 0:
                    print(f"\n      Active contacts:")
                    for j in range(min(5, env.data.ncon)):
                        contact = env.data.contact[j]
                        name1 = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom_{contact.geom1}"
                        name2 = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom_{contact.geom2}"

                        # Check if this is fingertip-paper
                        is_fingertip = (contact.geom1 in [env.fixed_fingertip_id, env.moving_fingertip_id] or
                                        contact.geom2 in [env.fixed_fingertip_id, env.moving_fingertip_id])
                        is_paper = (env.model.geom_bodyid[contact.geom1] == env.paper_body_id or
                                    env.model.geom_bodyid[contact.geom2] == env.paper_body_id)

                        marker = ""
                        if is_fingertip and is_paper:
                            marker = " ✅ FINGERTIP-PAPER!"
                        elif is_paper and not is_fingertip:
                            marker = " ⚠️  ROBOT-PAPER (not fingertip)"

                        print(f"         [{j}] {name1} ({contact.geom1}) <-> {name2} ({contact.geom2}){marker}")

        if info.get('fingertip_contact_reward', 0.0) > 0:
            print(f"\n   ✅ SUCCESS: Fingertip-paper contact reward is working!")
        else:
            print(f"\n   ❌ ISSUE: No fingertip reward despite manipulation")

    # =========================================================================
    # TEST 2: Move robot arm INTO paper
    # =========================================================================
    print("\n" + "="*80)
    print("TEST 2: Move Arm Into Paper (to trigger robot-paper penalty)")
    print("="*80)

    env.reset()
    paper_pos = env.data.xpos[env.paper_body_id].copy()
    print(f"\n📍 Paper at: ({paper_pos[0]:.3f}, {paper_pos[1]:.3f}, {paper_pos[2]:.3f})")

    # Move wrist body directly to paper position
    wrist_body_id = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_BODY, "wrist")

    # Manipulate joints to move wrist toward paper
    # (we can't directly set body position, only joint positions)
    qpos = env.data.qpos.copy()
    qpos[env.joint_ids[0]] = 0.0  # Pan
    qpos[env.joint_ids[1]] = 0.5  # Lift
    qpos[env.joint_ids[2]] = -1.0  # Elbow
    env.data.qpos[:] = qpos
    mj.mj_forward(env.model, env.data)

    wrist_pos = env.data.xpos[wrist_body_id].copy()
    print(f"   Wrist moved to: ({wrist_pos[0]:.3f}, {wrist_pos[1]:.3f}, {wrist_pos[2]:.3f})")

    # Now move paper UP to intersect with wrist
    paper_qpos_start = 7
    env.data.qpos[paper_qpos_start] = wrist_pos[0]
    env.data.qpos[paper_qpos_start + 1] = wrist_pos[1]
    env.data.qpos[paper_qpos_start + 2] = wrist_pos[2]  # Same Z as wrist
    mj.mj_forward(env.model, env.data)

    print(f"   Paper moved to: ({env.data.qpos[paper_qpos_start]:.3f}, "
          f"{env.data.qpos[paper_qpos_start+1]:.3f}, {env.data.qpos[paper_qpos_start+2]:.3f})")

    # Step to let contacts form
    action = np.zeros(6)
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\n   📊 After step:")
    print(f"      Contacts detected: {env.data.ncon}")
    print(f"      Robot-paper penalty: {info.get('unwanted_paper_contact_penalty', 0.0):+.3f}")
    print(f"      Total reward: {info.get('total_reward', 0.0):+.3f}")

    if info.get('unwanted_paper_contact_penalty', 0.0) < 0:
        print(f"\n   ✅ SUCCESS: Robot-paper penalty is working!")
    else:
        print(f"\n   ❌ ISSUE: No robot-paper penalty despite collision")

    # Show contacts
    if env.data.ncon > 0:
        print(f"\n   Active contacts:")
        for j in range(min(5, env.data.ncon)):
            contact = env.data.contact[j]
            name1 = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom_{contact.geom1}"
            name2 = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom_{contact.geom2}"
            print(f"      [{j}] {name1} ({contact.geom1}) <-> {name2} ({contact.geom2})")

    env.close()

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    force_contact_test()
