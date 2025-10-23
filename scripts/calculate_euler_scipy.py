#!/usr/bin/env python3
"""
Calculate correct euler angles using scipy's Rotation class.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


print("="*80)
print("Correct Euler Angle Calculation (Using Scipy)")
print("="*80)

# Desired camera forward direction (perpendicular to mounting hole, inward)
desired_forward = np.array([-0.2276068, 0.88252636, -0.41151229])
desired_forward = desired_forward / np.linalg.norm(desired_forward)

print(f"\nDesired camera forward direction (normalized): {desired_forward}")

# Camera's local forward is -Z axis: [0, 0, -1]
# We need a rotation that maps [0, 0, -1] to desired_forward

# Method: Find rotation that aligns -Z with desired_forward
local_forward = np.array([0, 0, -1])

# Calculate rotation axis (cross product)
rotation_axis = np.cross(local_forward, desired_forward)
rotation_axis_norm = np.linalg.norm(rotation_axis)

if rotation_axis_norm < 1e-6:
    # Vectors are parallel or anti-parallel
    if np.dot(local_forward, desired_forward) > 0:
        # Already aligned
        rotation = R.from_euler('xyz', [0, 0, 0])
    else:
        # Need 180° rotation around any perpendicular axis
        rotation = R.from_euler('xyz', [np.pi, 0, 0])
else:
    # Normal case: use axis-angle rotation
    rotation_axis = rotation_axis / rotation_axis_norm
    angle = np.arccos(np.clip(np.dot(local_forward, desired_forward), -1, 1))

    # Create rotation from axis-angle
    rotation = R.from_rotvec(rotation_axis * angle)

# Convert to euler angles (xyz convention = roll, pitch, yaw)
# MuJoCo uses (yaw, pitch, roll) convention which is (z, y, x) in scipy
euler_xyz = rotation.as_euler('xyz', degrees=False)  # [roll, pitch, yaw]
euler_zyx = rotation.as_euler('zyx', degrees=False)  # [yaw, pitch, roll]

print(f"\nEuler angles (xyz / roll-pitch-yaw):")
print(f"  Roll: {euler_xyz[0]:.4f} rad = {np.degrees(euler_xyz[0]):.1f}°")
print(f"  Pitch: {euler_xyz[1]:.4f} rad = {np.degrees(euler_xyz[1]):.1f}°")
print(f"  Yaw: {euler_xyz[2]:.4f} rad = {np.degrees(euler_xyz[2]):.1f}°")

print(f"\nEuler angles (zyx / yaw-pitch-roll) - MuJoCo format:")
print(f"  Yaw: {euler_zyx[0]:.4f} rad = {np.degrees(euler_zyx[0]):.1f}°")
print(f"  Pitch: {euler_zyx[1]:.4f} rad = {np.degrees(euler_zyx[1]):.1f}°")
print(f"  Roll: {euler_zyx[2]:.4f} rad = {np.degrees(euler_zyx[2]):.1f}°")

# Verify the calculation
calculated_forward = rotation.apply(local_forward)

print(f"\nVerification:")
print(f"  Calculated forward from rotation: {calculated_forward}")
print(f"  Expected forward: {desired_forward}")

# Check error
error = np.linalg.norm(calculated_forward - desired_forward)
angle_error = np.arccos(np.clip(np.dot(calculated_forward, desired_forward), -1, 1))

print(f"  Vector error: {error:.9f}")
print(f"  Angular error: {np.degrees(angle_error):.6f}°")

if angle_error < 0.01:  # Less than 0.5 degrees
    print(f"\n✅ Euler angles are correct!")
else:
    print(f"\n❌ Euler angles have error: {np.degrees(angle_error):.1f}°")

print("\n" + "="*80)
print("FINAL RECOMMENDATION (MuJoCo format: yaw pitch roll)")
print("="*80)
print(f"\n<camera name=\"wrist_camera\"")
print(f"        pos=\"0.0024 0.0783 0.0013\"")
print(f"        euler=\"{euler_zyx[0]:.4f} {euler_zyx[1]:.4f} {euler_zyx[2]:.4f}\"")
print(f"        fovy=\"140\"/>")
print("\n" + "="*80)
