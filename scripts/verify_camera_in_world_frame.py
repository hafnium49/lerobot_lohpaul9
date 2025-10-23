#!/usr/bin/env python3
"""
Verify the camera position by loading the MuJoCo model and computing world frame position.
"""

import mujoco
import numpy as np


print("="*80)
print("Verifying Camera Position in World Frame")
print("="*80)

# Load the MuJoCo model
xml_path = "src/lerobot/envs/so101_assets/paper_square_realistic.xml"
print(f"\nLoading: {xml_path}")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Reset to home position
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

# Find camera
camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")

if camera_id == -1:
    print("❌ Camera 'wrist_camera' not found!")
    exit(1)

print(f"✅ Found camera 'wrist_camera' (id={camera_id})")

# Get camera position in world frame
camera_pos_world = data.cam_xpos[camera_id].copy()

print(f"\nCamera Position (World frame from MuJoCo):")
print(f"  [{camera_pos_world[0]:.3f}, {camera_pos_world[1]:.3f}, {camera_pos_world[2]:.3f}] m")

# Get camera orientation matrix in world frame
camera_mat_world = data.cam_xmat[camera_id].reshape(3, 3).copy()

# Camera forward direction (camera looks along -Z axis in its local frame)
local_forward = np.array([0, 0, -1])
camera_forward_world = camera_mat_world @ local_forward

print(f"\nCamera Forward Direction (World frame):")
print(f"  [{camera_forward_world[0]:.3f}, {camera_forward_world[1]:.3f}, {camera_forward_world[2]:.3f}]")

# Get camera configuration from XML
print(f"\n" + "="*80)
print("Camera Configuration in XML (Gripper frame)")
print("="*80)

# Find the camera in the model
cam_pos_body = model.cam_pos[camera_id].copy()
cam_quat_body = model.cam_quat[camera_id].copy()

print(f"\nCamera relative to parent body:")
print(f"  Position: [{cam_pos_body[0]:.4f}, {cam_pos_body[1]:.4f}, {cam_pos_body[2]:.4f}] m")
print(f"  Quaternion: [{cam_quat_body[0]:.6f}, {cam_quat_body[1]:.6f}, {cam_quat_body[2]:.6f}, {cam_quat_body[3]:.6f}]")

# Convert quaternion to axis-angle for comparison
from scipy.spatial.transform import Rotation as R

rot = R.from_quat([cam_quat_body[1], cam_quat_body[2], cam_quat_body[3], cam_quat_body[0]])  # scipy uses [x,y,z,w]
rotvec = rot.as_rotvec()
angle = np.linalg.norm(rotvec)
if angle > 1e-6:
    axis = rotvec / angle
else:
    axis = np.array([1, 0, 0])
    angle = 0

print(f"\nAxis-angle representation:")
print(f"  Axis: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
print(f"  Angle: {angle:.6f} rad = {np.degrees(angle):.1f}°")

print(f"\n" + "="*80)
print("Expected vs Actual Comparison")
print("="*80)

# Expected from our calculation
expected_pos_gripper = np.array([0.0024, 0.0609, 0.0054])
expected_axis = np.array([1.0, 0.0, 0.0])
expected_angle = 1.134477

print(f"\nExpected (from screw hole analysis):")
print(f"  Position (gripper):  [{expected_pos_gripper[0]:.4f}, {expected_pos_gripper[1]:.4f}, {expected_pos_gripper[2]:.4f}] m")
print(f"  Axis-angle: [{expected_axis[0]:.6f}, {expected_axis[1]:.6f}, {expected_axis[2]:.6f}] {expected_angle:.6f} rad")

print(f"\nActual (from MuJoCo model):")
print(f"  Position (gripper):  [{cam_pos_body[0]:.4f}, {cam_pos_body[1]:.4f}, {cam_pos_body[2]:.4f}] m")
print(f"  Axis-angle: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}] {angle:.6f} rad")

# Check if they match
pos_diff = np.linalg.norm(cam_pos_body - expected_pos_gripper)
angle_diff = abs(angle - expected_angle)

print(f"\nDifference:")
print(f"  Position error: {pos_diff*1000:.3f} mm")
print(f"  Angle error: {np.degrees(angle_diff):.3f}°")

if pos_diff < 0.001 and angle_diff < 0.01:  # 1mm and 0.01 rad tolerance
    print(f"\n✅ Configuration MATCHES! Camera is correctly configured.")
else:
    print(f"\n⚠️  Configuration MISMATCH detected!")
    print(f"\n  Position should be: {expected_pos_gripper}")
    print(f"  Position is:        {cam_pos_body}")
    print(f"  Orientation should be: axis={expected_axis}, angle={expected_angle:.6f}")
    print(f"  Orientation is:        axis={axis}, angle={angle:.6f}")

print("="*80)
