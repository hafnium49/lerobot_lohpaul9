#!/usr/bin/env python3
"""
Analyze the camera mount geometry to determine correct camera position.
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
    print("Camera Mount Position Analysis")
    print("="*80)

    # Create environment
    env = SO101ResidualEnv(render_mode="rgb_array")
    obs, info = env.reset()

    model = env.model
    data = env.data

    # Get gripper body ID
    gripper_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "gripper")
    print(f"\nGripper body ID: {gripper_body_id}")
    print(f"Gripper body position: {data.xpos[gripper_body_id]}")
    print(f"Gripper body rotation:\n{data.xmat[gripper_body_id].reshape(3, 3)}")

    # Get camera mount geometry
    print("\n" + "="*80)
    print("Camera Mount Mesh Info")
    print("="*80)

    # Find the camera mount geom
    for i in range(model.ngeom):
        geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
        if geom_name and "Wrist_Cam_Mount" in geom_name:
            print(f"\nGeom: {geom_name}")
            print(f"  Type: {model.geom_type[i]}")
            print(f"  Body ID: {model.geom_bodyid[i]}")
            body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[i])
            print(f"  Attached to body: {body_name}")
            print(f"  Position (local): {model.geom_pos[i]}")
            print(f"  Position (world): {data.geom_xpos[i]}")
            print(f"  Size: {model.geom_size[i]}")

    # Get camera information
    print("\n" + "="*80)
    print("Camera Information")
    print("="*80)

    wrist_cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    print(f"\nWrist camera ID: {wrist_cam_id}")
    print(f"Wrist camera body: {model.cam_bodyid[wrist_cam_id]}")
    body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.cam_bodyid[wrist_cam_id])
    print(f"Attached to body: {body_name}")
    print(f"Camera position (local): {model.cam_pos[wrist_cam_id]}")
    print(f"Camera position (world): {data.cam_xpos[wrist_cam_id]}")
    print(f"Camera rotation:\n{data.cam_xmat[wrist_cam_id].reshape(3, 3)}")
    forward = -data.cam_xmat[wrist_cam_id].reshape(3, 3)[:, 2]
    print(f"Camera forward direction: {forward}")

    # Camera mount STL bounds (from earlier analysis)
    print("\n" + "="*80)
    print("Camera Mount STL Bounds (in meters, after 0.001 scale)")
    print("="*80)
    print("  X: [-0.0352, 0.0300] m  (width: 0.0652 m = 65.2 mm)")
    print("  Y: [-0.0894, 0.0242] m  (depth: 0.1136 m = 113.6 mm)")
    print("  Z: [-0.0148, 0.1054] m  (height: 0.1202 m = 120.2 mm)")
    print("\nThe camera mounting bracket should be on the extension arm")
    print("which extends in the -Y direction (toward workspace)")

    # Recommended camera position
    print("\n" + "="*80)
    print("Recommended Camera Position")
    print("="*80)
    print("\nBased on the camera mount geometry:")
    print("  - Extension arm extends ~90mm in -Y direction")
    print("  - Camera should be mounted at the end of the arm")
    print("  - Height should be around the middle of the mount (~50mm from base)")
    print("\nRecommended local position (in gripper frame):")
    print("  pos: 0.0 -0.090 0.050")
    print("  euler: 0 0.3 0  (pitched down to see workspace)")

    env.close()

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
