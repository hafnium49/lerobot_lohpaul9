#!/usr/bin/env python3
"""
Test MuJoCo's euler angle convention and find correct angles.
"""

import sys
from pathlib import Path

import mujoco as mj
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.envs.so101_residual_env import SO101ResidualEnv


print("="*80)
print("MuJoCo Euler Convention Test")
print("="*80)

# Create environment
env = SO101ResidualEnv(render_mode="rgb_array")
obs, info = env.reset()

model = env.model
data = env.data

# Get camera info
wrist_cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "wrist_camera")

print(f"\nWrist camera ID: {wrist_cam_id}")
print(f"Camera position (local): {model.cam_pos[wrist_cam_id]}")
print(f"Camera quat (local): {model.cam_quat[wrist_cam_id]}")  # MuJoCo stores quaternion

# Get camera forward direction from data
forward = -data.cam_xmat[wrist_cam_id].reshape(3, 3)[:, 2]
print(f"Camera forward (from xmat): {forward}")

# Expected forward
expected_forward = np.array([-0.2276068, 0.88252636, -0.41151229])
expected_forward = expected_forward / np.linalg.norm(expected_forward)
print(f"\nExpected forward (perpendicular to mount): {expected_forward}")

# Calculate quaternion that would give us expected forward
# We need to rotate from [0, 0, -1] to expected_forward

local_forward = np.array([0, 0, -1])
rotation_axis = np.cross(local_forward, expected_forward)
rotation_axis_norm = np.linalg.norm(rotation_axis)

if rotation_axis_norm < 1e-6:
    if np.dot(local_forward, expected_forward) > 0:
        quat = np.array([1, 0, 0, 0])  # No rotation
    else:
        quat = np.array([0, 1, 0, 0])  # 180° around X
else:
    rotation_axis = rotation_axis / rotation_axis_norm
    angle = np.arccos(np.clip(np.dot(local_forward, expected_forward), -1, 1))

    # Axis-angle to quaternion
    half_angle = angle / 2
    quat = np.array([
        np.cos(half_angle),
        rotation_axis[0] * np.sin(half_angle),
        rotation_axis[1] * np.sin(half_angle),
        rotation_axis[2] * np.sin(half_angle)
    ])

print(f"\nCalculated quaternion for desired orientation: {quat}")

# Convert quaternion to euler using MuJoCo
euler = np.zeros(3)
mj.mju_quat2Vel(euler, quat, 1.0)  # This converts quat to axis-angle

print(f"MuJoCo axis-angle from quat: {euler}")

# Better: use mat2Euler
mat = np.zeros(9)
mj.mju_quat2Mat(mat, quat)
euler_xyz = np.zeros(3)
mj.mju_mat2Euler(euler_xyz, mat)  # xyz convention

print(f"MuJoCo euler (xyz): {euler_xyz}")
print(f"  In degrees: [{np.degrees(euler_xyz[0]):.1f}, {np.degrees(euler_xyz[1]):.1f}, {np.degrees(euler_xyz[2]):.1f}]")

env.close()

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print(f"\n<camera name=\"wrist_camera\"")
print(f"        pos=\"0.0024 0.0783 0.0013\"")
print(f"        euler=\"{euler_xyz[0]:.4f} {euler_xyz[1]:.4f} {euler_xyz[2]:.4f}\"")
print(f"        fovy=\"140\"/>")
print("\n" + "="*80)
