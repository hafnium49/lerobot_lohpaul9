#!/usr/bin/env python
"""
Compare collision geometry vs visual geometry positions for robot bodies.
"""

import sys
sys.path.insert(0, 'src')
import mujoco as mj

# Load model
model = mj.MjModel.from_xml_path('src/lerobot/envs/so101_assets/paper_square_realistic.xml')
data = mj.MjData(model)

# Set to home position
joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
home_pos = [0.0, 0.3, -0.6, 0.0, 0.0, 0.0]

for i, joint_name in enumerate(joint_names):
    try:
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            data.qpos[model.jnt_qposadr[joint_id]] = home_pos[i]
    except:
        pass

# Forward kinematics
mj.mj_forward(model, data)

print("="*80)
print("COLLISION GEOMETRY vs VISUAL GEOMETRY COMPARISON")
print("="*80)

# Focus on bodies that might touch Z=0 based on video observation
bodies_at_table = ["lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"]

for body_name in bodies_at_table:
    try:
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        body_z = data.xpos[body_id][2]

        print(f"\n{body_name.upper()}:")
        print(f"  Body center: Z = {body_z:.4f}m")

        # Find collision geoms for this body
        collision_geoms = []
        visual_geoms = []

        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] == body_id:
                geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
                geom_group = model.geom_group[geom_id]
                geom_type = model.geom_type[geom_id]
                geom_size = model.geom_size[geom_id]

                # Geom position in world frame
                geom_z = data.geom_xpos[geom_id][2]

                geom_contype = model.geom_contype[geom_id]
                geom_conaffinity = model.geom_conaffinity[geom_id]

                # Group 0 with contype=1 = collision
                # Group 2 with contype=0 = visual only
                if geom_group == 0 and geom_contype > 0:
                    collision_geoms.append((geom_name, geom_type, geom_size, geom_z))
                elif geom_group == 2:
                    visual_geoms.append((geom_name, geom_type, geom_size, geom_z))

        if collision_geoms:
            print(f"  Collision geoms (group 0):")
            for name, gtype, size, z in collision_geoms:
                type_names = ["sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh"]
                type_name = type_names[gtype] if gtype < len(type_names) else f"type_{gtype}"
                print(f"    {name}: type={type_name}, Z={z:.4f}m")
                if gtype == 5 and len(size) >= 3:  # mesh
                    # For mesh, size is bounding box - check if it extends below Z=0
                    min_z = z - size[2]
                    print(f"      → Mesh bottom (approx): Z = {min_z:.4f}m")

        if visual_geoms:
            print(f"  Visual geoms (group 2):")
            for name, gtype, size, z in visual_geoms:
                type_names = ["sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh"]
                type_name = type_names[gtype] if gtype < len(type_names) else f"type_{gtype}"
                print(f"    {name}: type={type_name}, Z={z:.4f}m")
                if gtype == 5 and len(size) >= 3:  # mesh
                    min_z = z - size[2]
                    print(f"      → Mesh bottom (approx): Z = {min_z:.4f}m")

    except Exception as e:
        print(f"\nError processing {body_name}: {e}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("If visual mesh bottoms are below Z=0 but collision geom bottoms are at/above Z=0,")
print("then the 'penetration' you see in videos is purely visual - physics is correct!")
print("="*80)
