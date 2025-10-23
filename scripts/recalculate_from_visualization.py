#!/usr/bin/env python3
"""
Recalculate camera position and orientation from the values shown in
complete_camera_mount_visualization.png
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


print("="*80)
print("Recalculating Camera Configuration from Visualization")
print("="*80)

# Values directly from complete_camera_mount_visualization.png summary
print("\nFrom complete_camera_mount_visualization.png:")
print("\n4 Corner Screw Holes (STL frame):")

screw_holes_mm = np.array([
    [-12.4, -61.1, -10.7],  # H1 (red)
    [-13.1, -85.8,   1.9],  # H4 (green)
    [ 17.2, -61.6, -11.0],  # H6 (blue)
    [ 18.1, -85.9,   1.9],  # H9 (orange)
])

screw_holes_m = screw_holes_mm / 1000.0  # Convert to meters

labels = ["H1 (red)", "H4 (green)", "H6 (blue)", "H9 (orange)"]
for hole_mm, hole_m, label in zip(screw_holes_mm, screw_holes_m, labels):
    print(f"  {label:15s}: [{hole_m[0]:7.4f}, {hole_m[1]:7.4f}, {hole_m[2]:7.4f}] m")
    print(f"                   [{hole_mm[0]:6.1f}, {hole_mm[1]:6.1f}, {hole_mm[2]:6.1f}] mm")

# Camera position from visualization
camera_stl_mm = np.array([2.4, -61.1, -4.5])
camera_stl_m = camera_stl_mm / 1000.0

print(f"\nCamera Position (STL frame) from visualization:")
print(f"  [{camera_stl_m[0]:.4f}, {camera_stl_m[1]:.4f}, {camera_stl_m[2]:.4f}] m")
print(f"  [{camera_stl_mm[0]:.1f}, {camera_stl_mm[1]:.1f}, {camera_stl_mm[2]:.1f}] mm")

# Transform to gripper frame
# Camera mount: pos=[0, -0.000218214, 0.000949706], quat=[0, 1, 0, 0]
# quat [0,1,0,0] = 180° rotation around Y-axis
# Transformation: (x,y,z) -> (x, -y, -z) + mount_pos

mount_pos = np.array([0.0, -0.000218214, 0.000949706])

print(f"\nCamera Mount Transformation:")
print(f"  Position: [{mount_pos[0]:.9f}, {mount_pos[1]:.9f}, {mount_pos[2]:.9f}] m")
print(f"  Rotation: quat=[0, 1, 0, 0] (180° around Y-axis)")
print(f"  Formula: (x, y, z) -> (x, -y, -z) + mount_pos")

# Apply transformation
camera_rotated = np.array([camera_stl_m[0], -camera_stl_m[1], -camera_stl_m[2]])
camera_gripper = mount_pos + camera_rotated

print(f"\nCamera Position (Gripper frame):")
print(f"  [{camera_gripper[0]:.4f}, {camera_gripper[1]:.4f}, {camera_gripper[2]:.4f}] m")
print(f"  [{camera_gripper[0]*1000:.1f}, {camera_gripper[1]*1000:.1f}, {camera_gripper[2]*1000:.1f}] mm")

# Surface normal and orientation
surface_normal_stl = np.array([0.0, 0.9063, -0.4226])
surface_normal_stl = surface_normal_stl / np.linalg.norm(surface_normal_stl)

print(f"\nSurface Normal (STL frame):")
print(f"  [{surface_normal_stl[0]:.6f}, {surface_normal_stl[1]:.6f}, {surface_normal_stl[2]:.6f}]")

tilt_angle = np.degrees(np.arccos(abs(surface_normal_stl[1])))
print(f"  Tilt from Y-axis: {tilt_angle:.1f}°")

# Surface normal after 180° Y-rotation: (x, y, z) -> (x, -y, -z)
normal_gripper = np.array([surface_normal_stl[0], -surface_normal_stl[1], -surface_normal_stl[2]])

# Camera points INWARD (opposite of outward surface normal)
camera_forward = -normal_gripper

print(f"\nCamera Forward Direction (Gripper frame):")
print(f"  [{camera_forward[0]:.6f}, {camera_forward[1]:.6f}, {camera_forward[2]:.6f}]")

# Calculate orientation using scipy
local_forward = np.array([0, 0, -1])  # Camera's -Z axis in local frame

rotation_axis = np.cross(local_forward, camera_forward)
rotation_axis_norm = np.linalg.norm(rotation_axis)

if rotation_axis_norm < 1e-6:
    # Parallel or anti-parallel
    if np.dot(local_forward, camera_forward) > 0:
        rotation = R.from_euler('xyz', [0, 0, 0])
    else:
        rotation = R.from_euler('xyz', [np.pi, 0, 0])
else:
    rotation_axis = rotation_axis / rotation_axis_norm
    angle = np.arccos(np.clip(np.dot(local_forward, camera_forward), -1, 1))
    rotation = R.from_rotvec(rotation_axis * angle)

# Get axis-angle representation
rotvec = rotation.as_rotvec()
angle_rad = np.linalg.norm(rotvec)
if angle_rad > 1e-6:
    axis = rotvec / angle_rad
else:
    axis = np.array([1, 0, 0])
    angle_rad = 0

print(f"\nOrientation (Axis-Angle):")
print(f"  Axis:  [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
print(f"  Angle: {angle_rad:.6f} rad = {np.degrees(angle_rad):.1f}°")

# Verify
calculated_forward = rotation.apply(local_forward)
angular_error = np.degrees(np.arccos(np.clip(np.dot(calculated_forward, camera_forward), -1, 1)))

print(f"\nVerification:")
print(f"  Calculated forward: [{calculated_forward[0]:.6f}, {calculated_forward[1]:.6f}, {calculated_forward[2]:.6f}]")
print(f"  Expected forward:   [{camera_forward[0]:.6f}, {camera_forward[1]:.6f}, {camera_forward[2]:.6f}]")
print(f"  Angular error: {angular_error:.6f}°")

if angular_error < 0.001:
    print(f"  ✅ Perfect alignment!")
else:
    print(f"  ⚠️  Small error detected")

# Final configuration
print("\n" + "="*80)
print("FINAL CAMERA CONFIGURATION FOR XML")
print("="*80)
print(f"\n<camera name=\"wrist_camera\"")
print(f"        pos=\"{camera_gripper[0]:.4f} {camera_gripper[1]:.4f} {camera_gripper[2]:.4f}\"")
print(f"        axisangle=\"{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f} {angle_rad:.6f}\"")
print(f"        fovy=\"140\"/>")

print("\n" + "="*80)
