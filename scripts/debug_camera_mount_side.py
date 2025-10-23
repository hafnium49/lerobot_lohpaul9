#!/usr/bin/env python3
"""
Debug script to understand which side the camera mount is on
and where the camera should be positioned.
"""

import sys
from pathlib import Path

import mujoco as mj
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.envs.so101_residual_env import SO101ResidualEnv


def main():
    print("="*80)
    print("Camera Mount Side Analysis")
    print("="*80)

    # Create environment
    env = SO101ResidualEnv(render_mode="rgb_array")
    obs, info = env.reset()

    model = env.model
    data = env.data

    # Get gripper body ID and transform
    gripper_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "gripper")
    gripper_pos = data.xpos[gripper_body_id].copy()
    gripper_rot = data.xmat[gripper_body_id].reshape(3, 3).copy()

    print(f"\nGripper body position (world): {gripper_pos}")
    print(f"Gripper rotation matrix:\n{gripper_rot}")

    # Get all geoms attached to gripper body
    print("\n" + "="*80)
    print("Geometries attached to gripper body:")
    print("="*80)

    for i in range(model.ngeom):
        if model.geom_bodyid[i] == gripper_body_id:
            geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
            geom_type = model.geom_type[i]
            local_pos = model.geom_pos[i]
            world_pos = data.geom_xpos[i]

            type_names = ["plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh"]
            type_name = type_names[geom_type] if geom_type < len(type_names) else f"type_{geom_type}"

            print(f"\nGeom: {geom_name}")
            print(f"  Type: {type_name}")
            print(f"  Local pos: {local_pos}")
            print(f"  World pos: {world_pos}")

            if "Cam_Mount" in str(geom_name):
                print(f"  >>> CAMERA MOUNT FOUND <<<")

                # The camera mount mesh origin in gripper frame
                print(f"\n  Camera mount is positioned at: {local_pos} in gripper frame")

                # Transform local position to understand orientation
                # The camera mount STL has these bounds:
                # X: [-0.0352, 0.0300] m (width: 65.2 mm)
                # Y: [-0.0894, 0.0242] m (depth: 113.6 mm) - extension arm in -Y
                # Z: [-0.0148, 0.1054] m (height: 120.2 mm)

                print(f"\n  STL local bounds (before placement):")
                print(f"    X: [-0.0352, 0.0300] m (width: 65.2 mm)")
                print(f"    Y: [-0.0894, 0.0242] m (extension arm in -Y)")
                print(f"    Z: [-0.0148, 0.1054] m (height: 120.2 mm)")

                # Camera should be at the end of extension arm
                # Extension arm is at Y = -0.0894 (minimum Y)
                # Camera at mid-height: Z = 0.050 (approximately)
                # Camera at center X: X = -0.001 (approximately centered)

                camera_local_on_mount = np.array([-0.001, -0.0894, 0.050])

                # Transform to gripper frame
                camera_in_gripper = local_pos + camera_local_on_mount

                print(f"\n  Recommended camera position (gripper frame):")
                print(f"    Mount position: {local_pos}")
                print(f"    + Camera offset on mount: {camera_local_on_mount}")
                print(f"    = Camera position: {camera_in_gripper}")

    # Check current camera position
    print("\n" + "="*80)
    print("Current Camera Position:")
    print("="*80)

    wrist_cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    cam_local_pos = model.cam_pos[wrist_cam_id]
    cam_world_pos = data.cam_xpos[wrist_cam_id]

    print(f"\nWrist camera:")
    print(f"  Local position (gripper frame): {cam_local_pos}")
    print(f"  World position: {cam_world_pos}")

    # Also check moving jaw position for reference
    print("\n" + "="*80)
    print("Moving Jaw Position (for reference):")
    print("="*80)

    moving_jaw_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
    if moving_jaw_body_id >= 0:
        moving_jaw_pos = data.xpos[moving_jaw_body_id]
        print(f"Moving jaw world position: {moving_jaw_pos}")

        # Get moving jaw's local position relative to gripper
        for i in range(model.ngeom):
            geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
            if geom_name and "moving_jaw" in geom_name:
                body_id = model.geom_bodyid[i]
                if body_id == moving_jaw_body_id:
                    print(f"Moving jaw geom local pos: {model.geom_pos[i]}")
                    print(f"Moving jaw geom world pos: {data.geom_xpos[i]}")
                    break

    env.close()
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
