#!/usr/bin/env python3
"""
Calculate wrist camera orientation with 90° CLOCKWISE rotation.
This rotates the camera view 90° CW around its viewing direction (roll).
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


# Original wrist camera orientation (before any CCW rotation)
original_axis = np.array([-0.999821, -0.018912, -0.000000])
original_angle = 0.479911  # radians

print("="*80)
print("Wrist Camera Rotation Calculation (90° CLOCKWISE)")
print("="*80)

print("\nOriginal orientation (axis-angle):")
print(f"  Axis:  [{original_axis[0]:.6f}, {original_axis[1]:.6f}, {original_axis[2]:.6f}]")
print(f"  Angle: {original_angle:.6f} rad = {np.degrees(original_angle):.2f}°")

# Convert original orientation to rotation matrix
original_rotation = R.from_rotvec(original_axis * original_angle)

print("\nOriginal camera orientation:")
print(f"  Rotation matrix:\n{original_rotation.as_matrix()}")

# Camera local axes after original rotation
camera_forward = original_rotation.apply([0, 0, -1])  # -Z in camera frame
camera_right = original_rotation.apply([1, 0, 0])     # +X in camera frame
camera_up = original_rotation.apply([0, 1, 0])        # +Y in camera frame

print(f"\n  Forward (-Z): [{camera_forward[0]:.4f}, {camera_forward[1]:.4f}, {camera_forward[2]:.4f}]")
print(f"  Right   (+X): [{camera_right[0]:.4f}, {camera_right[1]:.4f}, {camera_right[2]:.4f}]")
print(f"  Up      (+Y): [{camera_up[0]:.4f}, {camera_up[1]:.4f}, {camera_up[2]:.4f}]")

# 90° CLOCKWISE rotation = -90° rotation around viewing direction (forward axis)
additional_rotation_angle = -np.pi / 2  # -90 degrees in radians (CW)
additional_rotation = R.from_rotvec(camera_forward * additional_rotation_angle)

print("\n" + "="*80)
print("Additional 90° CLOCKWISE Rotation (Roll)")
print("="*80)

print(f"\nRotation axis (camera forward): [{camera_forward[0]:.6f}, {camera_forward[1]:.6f}, {camera_forward[2]:.6f}]")
print(f"Rotation angle: {additional_rotation_angle:.6f} rad = {np.degrees(additional_rotation_angle):.2f}°")

# Compose rotations: new = additional * original
combined_rotation = additional_rotation * original_rotation

print("\n" + "="*80)
print("Combined Rotation")
print("="*80)

# Convert to axis-angle
combined_rotvec = combined_rotation.as_rotvec()
combined_angle = np.linalg.norm(combined_rotvec)

if combined_angle > 1e-6:
    combined_axis = combined_rotvec / combined_angle
else:
    combined_axis = np.array([1, 0, 0])
    combined_angle = 0.0

print("\nNew orientation (axis-angle):")
print(f"  Axis:  [{combined_axis[0]:.6f}, {combined_axis[1]:.6f}, {combined_axis[2]:.6f}]")
print(f"  Angle: {combined_angle:.6f} rad = {np.degrees(combined_angle):.2f}°")

# Verify the new camera axes
new_camera_forward = combined_rotation.apply([0, 0, -1])
new_camera_right = combined_rotation.apply([1, 0, 0])
new_camera_up = combined_rotation.apply([0, 1, 0])

print("\nNew camera orientation:")
print(f"  Forward (-Z): [{new_camera_forward[0]:.4f}, {new_camera_forward[1]:.4f}, {new_camera_forward[2]:.4f}]")
print(f"  Right   (+X): [{new_camera_right[0]:.4f}, {new_camera_right[1]:.4f}, {new_camera_right[2]:.4f}]")
print(f"  Up      (+Y): [{new_camera_up[0]:.4f}, {new_camera_up[1]:.4f}, {new_camera_up[2]:.4f}]")

# Verify forward direction hasn't changed
forward_error = np.linalg.norm(new_camera_forward - camera_forward)
print(f"\nForward direction error: {forward_error:.6e} (should be ~0)")

if forward_error < 1e-6:
    print("✓ Forward direction preserved (roll rotation correct)")
else:
    print("⚠ Warning: Forward direction changed!")

# Verify right and up have rotated 90° CW (right → down, up → right)
right_rotation_check = np.dot(new_camera_right, -camera_up)
up_rotation_check = np.dot(new_camera_up, camera_right)

print(f"\nRotation verification (90° CW):")
print(f"  new_right · (-old_up):  {right_rotation_check:.4f} (should be ~1)")
print(f"  new_up · old_right:     {up_rotation_check:.4f} (should be ~1)")

if abs(right_rotation_check - 1.0) < 0.01 and abs(up_rotation_check - 1.0) < 0.01:
    print("✓ 90° CW rotation verified")
else:
    print("⚠ Warning: Rotation may not be exactly 90° CW")

print("\n" + "="*80)
print("XML Configuration")
print("="*80)

print(f"\n<camera name=\"wrist_camera\"")
print(f"        pos=\"0.0025 0.0609 0.0120\"")
print(f"        axisangle=\"{combined_axis[0]:.6f} {combined_axis[1]:.6f} {combined_axis[2]:.6f} {combined_angle:.6f}\"")
print(f"        fovy=\"90\"/>")

print("\n" + "="*80)
