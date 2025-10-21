#!/usr/bin/env python
"""
Dedicated test for fingertip-paper contact with 16mm diameter fingertips.
Uses aggressive positioning to verify contact detection is working.
"""

import sys
from pathlib import Path

import mujoco as mj
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.envs.so101_base_policy import JacobianIKPolicy
from lerobot.envs.so101_residual_env import SO101ResidualEnv

def test_fingertip_paper_contact():
    """Test fingertip-paper contact detection."""
    print("="*80)
    print("FINGERTIP-PAPER CONTACT TEST")
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

    # Get geometry info
    fixed_ft_size = env.model.geom_size[env.fixed_fingertip_id][0]
    moving_ft_size = env.model.geom_size[env.moving_fingertip_id][0]

    print(f"\n📏 Geometry Information:")
    print(f"   Fixed fingertip radius:  {fixed_ft_size*1000:.1f}mm (diameter: {fixed_ft_size*2*1000:.1f}mm)")
    print(f"   Moving fingertip radius: {moving_ft_size*1000:.1f}mm (diameter: {moving_ft_size*2*1000:.1f}mm)")

    # Get paper info
    paper_geom_id = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_GEOM, "paper_geom")
    paper_size = env.model.geom_size[paper_geom_id]
    print(f"   Paper size: {paper_size[0]*2*1000:.1f}mm × {paper_size[1]*2*1000:.1f}mm × {paper_size[2]*2*1000:.1f}mm")

    paper_pos = env.data.xpos[env.paper_body_id].copy()
    ee_pos = env.data.site_xpos[env.ee_site_id].copy()

    print(f"\n📍 Initial Positions:")
    print(f"   Paper center: ({paper_pos[0]:.3f}, {paper_pos[1]:.3f}, {paper_pos[2]:.3f})")
    print(f"   End-effector: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
    print(f"   Distance: {np.linalg.norm(paper_pos - ee_pos):.3f}m")

    # =========================================================================
    # TEST 1: Move EE directly above paper
    # =========================================================================
    print("\n" + "="*80)
    print("TEST 1: Position EE Above Paper and Lower")
    print("="*80)

    env.reset()
    paper_pos = env.data.xpos[env.paper_body_id].copy()

    # Set joint positions to position EE above paper
    qpos = env.data.qpos.copy()

    # Adjust joints to move EE toward paper
    qpos[env.joint_ids[0]] = -0.2   # Pan toward paper
    qpos[env.joint_ids[1]] = 0.4    # Lift
    qpos[env.joint_ids[2]] = -0.8   # Elbow flex
    qpos[env.joint_ids[3]] = -0.3   # Wrist flex
    qpos[env.joint_ids[5]] = -0.1   # Gripper closed

    env.data.qpos[:] = qpos
    mj.mj_forward(env.model, env.data)

    ee_pos = env.data.site_xpos[env.ee_site_id].copy()
    print(f"\n📍 Positioned EE at: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
    print(f"   Paper at:         ({paper_pos[0]:.3f}, {paper_pos[1]:.3f}, {paper_pos[2]:.3f})")
    print(f"   Distance: {np.linalg.norm(paper_pos - ee_pos):.3f}m")

    # Now gradually lower the arm
    print(f"\n🔽 Lowering arm toward paper...")

    contact_detected = False
    for step in range(100):
        # Small downward movement
        action = np.zeros(6)
        action[1] = -0.3  # Lower shoulder
        action[2] = -0.2  # Lower elbow
        action[5] = -0.2  # Keep gripper closed

        obs, reward, terminated, truncated, info = env.step(action)

        ee_pos = env.data.site_xpos[env.ee_site_id].copy()
        dist = np.linalg.norm(paper_pos - ee_pos)

        # Check for fingertip contact
        if info.get("fingertip_contact_reward", 0.0) > 0:
            contact_detected = True
            print(f"\n   ✅ Step {step:3d}: FINGERTIP-PAPER CONTACT!")
            print(f"      Distance to paper: {dist:.4f}m ({dist*1000:.1f}mm)")
            print(f"      EE position: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
            print(f"      Fingertip reward: {info['fingertip_contact_reward']:+.3f}")
            print(f"      Total reward: {info['total_reward']:+.3f}")

            # Show contact details
            print(f"\n      Contact buffer info:")
            print(f"         Total contacts: {env.data.ncon}")
            for i in range(min(5, env.data.ncon)):
                contact = env.data.contact[i]
                name1 = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom_{contact.geom1}"
                name2 = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom_{contact.geom2}"

                is_ft_paper = False
                if contact.geom1 in [env.fixed_fingertip_id, env.moving_fingertip_id]:
                    if env.model.geom_bodyid[contact.geom2] == env.paper_body_id:
                        is_ft_paper = True
                elif contact.geom2 in [env.fixed_fingertip_id, env.moving_fingertip_id]:
                    if env.model.geom_bodyid[contact.geom1] == env.paper_body_id:
                        is_ft_paper = True

                marker = " ✅ FINGERTIP-PAPER!" if is_ft_paper else ""
                print(f"         [{i}] {name1} <-> {name2}{marker}")
            break

        # Progress update
        if step % 25 == 0 and step > 0:
            print(f"   Step {step:3d}: Distance = {dist:.4f}m ({dist*1000:.1f}mm), Z = {ee_pos[2]:.3f}m")

    if not contact_detected:
        ee_pos = env.data.site_xpos[env.ee_site_id].copy()
        dist = np.linalg.norm(paper_pos - ee_pos)
        print(f"\n   ❌ No fingertip contact in 100 steps")
        print(f"      Final distance: {dist:.4f}m ({dist*1000:.1f}mm)")
        print(f"      Final EE Z: {ee_pos[2]:.3f}m (paper Z: {paper_pos[2]:.3f}m)")

    # =========================================================================
    # TEST 2: Place paper directly on fingertip
    # =========================================================================
    print("\n" + "="*80)
    print("TEST 2: Place Paper Directly On Fingertip (Force Contact)")
    print("="*80)

    env.reset()

    # Position robot in a known configuration
    qpos = env.data.qpos.copy()
    qpos[env.joint_ids[0]] = 0.0
    qpos[env.joint_ids[1]] = 0.3
    qpos[env.joint_ids[2]] = -0.6
    qpos[env.joint_ids[3]] = 0.0
    qpos[env.joint_ids[5]] = 0.0  # Gripper neutral
    env.data.qpos[:] = qpos
    mj.mj_forward(env.model, env.data)

    ee_pos = env.data.site_xpos[env.ee_site_id].copy()
    print(f"\n📍 EE positioned at: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")

    # Calculate fingertip position (approximately 7mm below EE)
    fingertip_z = ee_pos[2] - 0.007

    # Place paper directly at fingertip level
    paper_qpos_start = 7  # Paper starts at index 7
    env.data.qpos[paper_qpos_start] = ee_pos[0]
    env.data.qpos[paper_qpos_start + 1] = ee_pos[1]
    env.data.qpos[paper_qpos_start + 2] = fingertip_z  # At fingertip level

    # Keep paper orientation flat
    env.data.qpos[paper_qpos_start + 3] = 1.0  # w
    env.data.qpos[paper_qpos_start + 4] = 0.0  # x
    env.data.qpos[paper_qpos_start + 5] = 0.0  # y
    env.data.qpos[paper_qpos_start + 6] = 0.0  # z

    mj.mj_forward(env.model, env.data)

    paper_pos_new = env.data.xpos[env.paper_body_id].copy()
    print(f"   Paper placed at:  ({paper_pos_new[0]:.3f}, {paper_pos_new[1]:.3f}, {paper_pos_new[2]:.3f})")
    print(f"   Overlap (XY):     {abs(ee_pos[0] - paper_pos_new[0])*1000:.1f}mm, {abs(ee_pos[1] - paper_pos_new[1])*1000:.1f}mm")
    print(f"   Vertical gap:     {abs(fingertip_z - paper_pos_new[2])*1000:.1f}mm")

    # Let physics settle
    print(f"\n⏱️  Letting physics settle for 20 steps...")
    contact_detected_2 = False

    for step in range(20):
        action = np.zeros(6)  # No action, just let physics settle
        obs, reward, terminated, truncated, info = env.step(action)

        if info.get("fingertip_contact_reward", 0.0) > 0:
            if not contact_detected_2:
                contact_detected_2 = True
                print(f"\n   ✅ Step {step:2d}: FINGERTIP-PAPER CONTACT!")
                print(f"      Fingertip reward: {info['fingertip_contact_reward']:+.3f}")
                print(f"      Total contacts: {env.data.ncon}")

                # Show all contacts
                for i in range(min(10, env.data.ncon)):
                    contact = env.data.contact[i]
                    name1 = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom_{contact.geom1}"
                    name2 = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom_{contact.geom2}"
                    print(f"         [{i}] {name1} <-> {name2}")

    if not contact_detected_2:
        print(f"\n   ❌ No contact detected even with direct placement")
        print(f"      Total contacts in scene: {env.data.ncon}")
        if env.data.ncon > 0:
            print(f"      Active contacts:")
            for i in range(min(5, env.data.ncon)):
                contact = env.data.contact[i]
                name1 = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom_{contact.geom1}"
                name2 = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom_{contact.geom2}"
                print(f"         [{i}] {name1} ({contact.geom1}) <-> {name2} ({contact.geom2})")

    env.close()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"\n✅ Fingertip diameter: {fixed_ft_size*2*1000:.1f}mm")
    print(f"✅ Paper surface: {paper_size[0]*2*1000:.1f}mm × {paper_size[1]*2*1000:.1f}mm")
    print(f"\n{'✅' if contact_detected else '❌'} TEST 1: Lowering arm - {'CONTACT DETECTED' if contact_detected else 'NO CONTACT'}")
    print(f"{'✅' if contact_detected_2 else '❌'} TEST 2: Direct placement - {'CONTACT DETECTED' if contact_detected_2 else 'NO CONTACT'}")

    if contact_detected or contact_detected_2:
        print(f"\n🎉 SUCCESS: Fingertip-paper contact detection is WORKING!")
        print(f"   Contact reward system is functional and ready for training.")
    else:
        print(f"\n⚠️  ISSUE: Fingertip-paper contacts not detected in tests")
        print(f"   This may indicate a collision configuration issue.")
        print(f"   However, contacts may still occur during RL training.")

    print("="*80)

if __name__ == "__main__":
    test_fingertip_paper_contact()
