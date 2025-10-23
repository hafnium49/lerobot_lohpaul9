#!/usr/bin/env python3
"""
Verify the current camera orientation and compare with mounting hole surface normal.
"""

import numpy as np


def euler_to_direction(yaw, pitch, roll=0):
    """
    Convert euler angles (ZYX convention) to forward direction vector.
    Camera looks along -Z in its local frame.
    """
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
    R = Rz @ Ry @ Rx

    # Camera looks along -Z in local frame
    forward_local = np.array([0, 0, -1])
    forward_global = R @ forward_local

    return forward_global


print("="*80)
print("Camera Orientation Verification")
print("="*80)

# Current camera euler angles from XML
current_euler = np.array([-0.2524, 0.4241, 0])
print(f"\nCurrent camera euler angles: {current_euler}")
print(f"  Yaw: {current_euler[0]:.4f} rad = {np.degrees(current_euler[0]):.1f}°")
print(f"  Pitch: {current_euler[1]:.4f} rad = {np.degrees(current_euler[1]):.1f}°")
print(f"  Roll: {current_euler[2]:.4f} rad = {np.degrees(current_euler[2]):.1f}°")

# Calculate current camera forward direction
current_forward = euler_to_direction(current_euler[0], current_euler[1], current_euler[2])
print(f"\nCurrent camera forward direction: {current_forward}")
print(f"  Normalized: {current_forward / np.linalg.norm(current_forward)}")

# Expected surface normal (perpendicular to mounting hole, pointing inward to workspace)
# From previous analysis: surface normal in gripper frame after rotation
surface_normal_outward = np.array([0.2276068, -0.88252636, 0.41151229])
surface_normal_inward = -surface_normal_outward  # Camera should point inward

print(f"\n" + "="*80)
print("Mounting Hole Surface Normal")
print("="*80)
print(f"\nSurface normal (outward from mount): {surface_normal_outward}")
print(f"Surface normal (inward, toward workspace): {surface_normal_inward}")
print(f"  Normalized: {surface_normal_inward / np.linalg.norm(surface_normal_inward)}")

# Compare directions
print(f"\n" + "="*80)
print("Comparison")
print("="*80)

dot_product = np.dot(current_forward, surface_normal_inward)
angle_diff = np.arccos(np.clip(dot_product, -1, 1))

print(f"\nDot product: {dot_product:.4f}")
print(f"Angle difference: {np.degrees(angle_diff):.1f}°")

if angle_diff < 0.01:  # Less than ~0.5 degrees
    print("\n✅ Camera IS oriented perpendicular to mounting hole surface")
else:
    print(f"\n❌ Camera is NOT perpendicular to mounting hole surface")
    print(f"   Current forward: {current_forward}")
    print(f"   Expected forward: {surface_normal_inward}")
    print(f"   Angular error: {np.degrees(angle_diff):.1f}°")

print("\n" + "="*80)
