#!/usr/bin/env python3
"""
Correctly calculate camera position accounting for mesh transform.
"""

import numpy as np


def quaternion_to_matrix(q):
    """Convert quaternion [w, x, y, z] to rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])


print("="*80)
print("Corrected Camera Position Calculation")
print("="*80)

# Camera mount mesh placement in gripper frame
mount_pos = np.array([0.0, -0.000218214, 0.000949706])
mount_quat = np.array([0, 1, 0, 0])  # [w, x, y, z] - 180° around Y-axis

print(f"\nCamera mount mesh in gripper frame:")
print(f"  Position: {mount_pos}")
print(f"  Quaternion: {mount_quat} (180° rotation around Y-axis)")

# Convert quaternion to rotation matrix
R = quaternion_to_matrix(mount_quat)
print(f"\nRotation matrix:")
print(R)

# Camera hole position in STL's local frame (from analysis)
camera_hole_stl = np.array([0.0024, -0.0785, -0.0004])
print(f"\nCamera hole in STL local frame:")
print(f"  Position: {camera_hole_stl}")

# Transform to gripper frame: pos_gripper = mount_pos + R @ pos_stl
camera_hole_gripper = mount_pos + R @ camera_hole_stl

print(f"\nCamera hole in gripper frame:")
print(f"  Position: {camera_hole_gripper}")
print(f"  Formatted: pos=\"{camera_hole_gripper[0]:.4f} {camera_hole_gripper[1]:.4f} {camera_hole_gripper[2]:.4f}\"")

# Surface normal in STL frame
surface_normal_stl = np.array([0.2276, 0.8825, -0.4115])
surface_normal_stl = surface_normal_stl / np.linalg.norm(surface_normal_stl)

print(f"\nSurface normal in STL frame:")
print(f"  Direction: {surface_normal_stl}")

# Transform normal to gripper frame
surface_normal_gripper = R @ surface_normal_stl

print(f"\nSurface normal in gripper frame:")
print(f"  Direction: {surface_normal_gripper}")

# Convert to euler angles
# Camera's -Z axis should point along the surface normal
# Camera frame: +X right, +Y down, -Z forward
# We need to find euler angles (ZYX convention) such that -Z points along surface_normal_gripper

# Simplified approach:
# If normal is [nx, ny, nz], and we want camera -Z to point that way
# Yaw (around Z): rotation in XY plane
yaw = np.arctan2(surface_normal_gripper[0], surface_normal_gripper[1])
# Pitch (around X): rotation from horizontal
pitch = np.arctan2(-surface_normal_gripper[2],
                   np.sqrt(surface_normal_gripper[0]**2 + surface_normal_gripper[1]**2))

print(f"\nEuler angles (yaw, pitch, roll):")
print(f"  Radians: [{yaw:.4f}, {pitch:.4f}, 0]")
print(f"  Degrees: [{np.degrees(yaw):.1f}°, {np.degrees(pitch):.1f}°, 0°]")
print(f"  Formatted: euler=\"{yaw:.4f} {pitch:.4f} 0\"")

print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)
print(f"\n<camera name=\"wrist_camera\"")
print(f"        pos=\"{camera_hole_gripper[0]:.4f} {camera_hole_gripper[1]:.4f} {camera_hole_gripper[2]:.4f}\"")
print(f"        euler=\"{yaw:.4f} {pitch:.4f} 0\"")
print(f"        fovy=\"140\"/>")
print("\n" + "="*80)
