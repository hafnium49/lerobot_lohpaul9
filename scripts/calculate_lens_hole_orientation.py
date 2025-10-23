#!/usr/bin/env python3
"""
Calculate camera orientation perpendicular to flat mounting surface.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


print("="*80)
print("Camera Orientation for Flat Mounting Surface")
print("="*80)

# Camera position at center of dodecagon lens hole (gripper frame)
camera_pos = np.array([0.0025, 0.0742, 0.0038])

# Surface normal (inward, toward workspace)
surface_normal = np.array([0, 0.9063, -0.4226])
surface_normal = surface_normal / np.linalg.norm(surface_normal)

print(f"\nCamera position (gripper frame): {camera_pos}")
print(f"Surface normal (inward): {surface_normal}")

# Camera should point along surface normal
desired_forward = surface_normal

# Camera's local forward is -Z axis: [0, 0, -1]
local_forward = np.array([0, 0, -1])

# Calculate rotation from local_forward to desired_forward
rotation_axis = np.cross(local_forward, desired_forward)
rotation_axis_norm = np.linalg.norm(rotation_axis)

if rotation_axis_norm < 1e-6:
    if np.dot(local_forward, desired_forward) > 0:
        rotation = R.from_euler('xyz', [0, 0, 0])
    else:
        rotation = R.from_euler('xyz', [np.pi, 0, 0])
else:
    rotation_axis = rotation_axis / rotation_axis_norm
    angle = np.arccos(np.clip(np.dot(local_forward, desired_forward), -1, 1))
    rotation = R.from_rotvec(rotation_axis * angle)

# Get axis-angle representation (what MuJoCo uses)
rotvec = rotation.as_rotvec()
angle_rad = np.linalg.norm(rotvec)
if angle_rad > 1e-6:
    axis = rotvec / angle_rad
else:
    axis = np.array([1, 0, 0])
    angle_rad = 0

print(f"\nAxis-angle representation:")
print(f"  Axis: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
print(f"  Angle: {angle_rad:.6f} rad = {np.degrees(angle_rad):.1f}°")

# MuJoCo axisangle format: "ax ay az angle"
print(f"\nMuJoCo axisangle format:")
print(f"  axisangle=\"{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f} {angle_rad:.6f}\"")

# Verify
calculated_forward = rotation.apply(local_forward)
print(f"\nVerification:")
print(f"  Calculated forward: {calculated_forward}")
print(f"  Expected forward: {desired_forward}")
error = np.linalg.norm(calculated_forward - desired_forward)
print(f"  Error: {error:.9f}")

print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)
print(f"\n<camera name=\"wrist_camera\"")
print(f"        pos=\"{camera_pos[0]:.4f} {camera_pos[1]:.4f} {camera_pos[2]:.4f}\"")
print(f"        axisangle=\"{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f} {angle_rad:.6f}\"")
print(f"        fovy=\"140\"/>")
print("\n" + "="*80)
