#!/usr/bin/env python3
"""
Calculate correct euler angles to make camera point along desired direction.
Uses proper direction-to-euler conversion.
"""

import numpy as np


def direction_to_euler(forward_direction):
    """
    Convert a forward direction vector to euler angles (yaw, pitch, roll).

    The camera looks along -Z in its local frame.
    We need to find euler angles such that R @ [0, 0, -1] = forward_direction

    Using ZYX euler convention:
    - Yaw (rotation around Z): determines azimuth in XY plane
    - Pitch (rotation around Y): determines elevation angle
    - Roll (rotation around X): rotation around forward axis (usually 0)
    """
    # Normalize the forward direction
    forward = forward_direction / np.linalg.norm(forward_direction)

    # For camera looking along -Z, we need:
    # After rotations, local -Z axis should point along 'forward'

    # Decompose forward direction
    fx, fy, fz = forward

    # Yaw: rotation around Z to align with XY projection
    # atan2(fx, fy) gives the yaw needed so that after yaw rotation,
    # the forward direction projects onto +Y in the rotated frame
    yaw = np.arctan2(fx, fy)

    # Pitch: rotation around (rotated) Y to tilt from horizontal
    # After yaw rotation, we need to pitch to align with the Z component
    # The horizontal distance is sqrt(fx^2 + fy^2)
    horizontal_dist = np.sqrt(fx**2 + fy**2)
    pitch = np.arctan2(-fz, horizontal_dist)

    # Roll: typically 0 for cameras
    roll = 0

    return yaw, pitch, roll


def euler_to_matrix(yaw, pitch, roll):
    """Convert euler angles to rotation matrix (ZYX convention)."""
    # Rotation matrices
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Combined rotation (ZYX order)
    return Rz @ Ry @ Rx


print("="*80)
print("Correct Euler Angle Calculation")
print("="*80)

# Desired camera forward direction (perpendicular to mounting hole, inward)
desired_forward = np.array([-0.2276068, 0.88252636, -0.41151229])

print(f"\nDesired camera forward direction: {desired_forward}")
print(f"  Normalized: {desired_forward / np.linalg.norm(desired_forward)}")

# Calculate euler angles
yaw, pitch, roll = direction_to_euler(desired_forward)

print(f"\nCalculated euler angles:")
print(f"  Yaw: {yaw:.4f} rad = {np.degrees(yaw):.1f}°")
print(f"  Pitch: {pitch:.4f} rad = {np.degrees(pitch):.1f}°")
print(f"  Roll: {roll:.4f} rad = {np.degrees(roll):.1f}°")

# Verify the calculation
R = euler_to_matrix(yaw, pitch, roll)
camera_local_forward = np.array([0, 0, -1])
calculated_forward = R @ camera_local_forward

print(f"\nVerification:")
print(f"  Calculated forward from euler: {calculated_forward}")
print(f"  Expected forward: {desired_forward / np.linalg.norm(desired_forward)}")

# Check error
error = np.linalg.norm(calculated_forward - desired_forward / np.linalg.norm(desired_forward))
angle_error = np.arccos(np.clip(np.dot(calculated_forward, desired_forward / np.linalg.norm(desired_forward)), -1, 1))

print(f"  Vector error: {error:.6f}")
print(f"  Angular error: {np.degrees(angle_error):.3f}°")

if angle_error < 0.01:  # Less than 0.5 degrees
    print(f"\n✅ Euler angles are correct!")
else:
    print(f"\n❌ Euler angles have error: {np.degrees(angle_error):.1f}°")

print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)
print(f"\n<camera name=\"wrist_camera\"")
print(f"        pos=\"0.0024 0.0783 0.0013\"")
print(f"        euler=\"{yaw:.4f} {pitch:.4f} {roll:.4f}\"")
print(f"        fovy=\"140\"/>")
print("\n" + "="*80)
