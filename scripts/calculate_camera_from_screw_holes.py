#!/usr/bin/env python3
"""
Calculate camera position from the 4 corner screw holes.
Based on analysis, the 4 corner screw holes are H1, H4, H6, H9.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


print("="*80)
print("Camera Position from 4 Corner Screw Holes")
print("="*80)

# From screw_holes_analysis.png, the 4 corner screw holes are:
# H1 (bottom-left): [-0.0124, -0.0611, -0.0107] m
# H4 (top-left): [-0.0131, -0.0858, 0.0019] m
# H6 (bottom-right): [0.0172, -0.0616, -0.0110] m
# H9 (top-right): [0.0181, -0.0859, 0.0019] m

screw_holes = np.array([
    [-0.0124, -0.0611, -0.0107],  # H1
    [-0.0131, -0.0858,  0.0019],  # H4
    [ 0.0172, -0.0616, -0.0110],  # H6
    [ 0.0181, -0.0859,  0.0019],  # H9
])

print("\n4 Corner Screw Holes (STL frame):")
labels = ["H1 (bottom-left)", "H4 (top-left)", "H6 (bottom-right)", "H9 (top-right)"]
for i, (hole, label) in enumerate(zip(screw_holes, labels)):
    print(f"  {label:20s}: [{hole[0]:7.4f}, {hole[1]:7.4f}, {hole[2]:7.4f}] m")

# Calculate geometric center (this is the camera position at lens hole center)
center_x = screw_holes[:, 0].mean()
center_y = screw_holes[:, 1].mean()
center_z = screw_holes[:, 2].mean()

print(f"\nGeometric Center (Camera Position in STL frame):")
print(f"  [{center_x:.4f}, {center_y:.4f}, {center_z:.4f}] m")

# Calculate mounting surface dimensions
x_span = screw_holes[:, 0].max() - screw_holes[:, 0].min()
y_span = screw_holes[:, 1].max() - screw_holes[:, 1].min()
z_span = screw_holes[:, 2].max() - screw_holes[:, 2].min()

print(f"\nScrew Hole Spacing:")
print(f"  X: {x_span*1000:.1f} mm (left-right)")
print(f"  Y: {y_span*1000:.1f} mm (depth)")
print(f"  Z: {z_span*1000:.1f} mm (up-down)")

# The Y-span shows the holes are at different depths
# Use the most outward Y position (maximum Y) as the surface Y
y_surface = screw_holes[:, 1].max()
print(f"\nMounting Surface Y (most outward): {y_surface:.4f} m")

# Update camera Y to be at the surface
center_y_surface = y_surface

print(f"\nCamera Position at Surface (STL frame):")
print(f"  [{center_x:.4f}, {center_y_surface:.4f}, {center_z:.4f}] m")

# Transform to gripper frame
# Camera mount: pos=[0, -0.000218214, 0.000949706], quat=[0, 1, 0, 0]
# quat [0,1,0,0] = 180° rotation around Y-axis
# Transformation: (x,y,z) -> (x, -y, -z) + mount_pos

mount_pos = np.array([0.0, -0.000218214, 0.000949706])
camera_stl = np.array([center_x, center_y_surface, center_z])
camera_rotated = np.array([camera_stl[0], -camera_stl[1], -camera_stl[2]])
camera_gripper = mount_pos + camera_rotated

print(f"\nCamera Position (Gripper frame):")
print(f"  [{camera_gripper[0]:.4f}, {camera_gripper[1]:.4f}, {camera_gripper[2]:.4f}] m")

# Surface normal - we determined this is [0, 0.9063, -0.4226] from earlier analysis
surface_normal_stl = np.array([0.0, 0.9063, -0.4226])
surface_normal_stl = surface_normal_stl / np.linalg.norm(surface_normal_stl)

print(f"\nSurface Normal (STL): {surface_normal_stl}")

# Surface normal after rotation: (x, y, z) -> (x, -y, -z)
normal_rotated = np.array([surface_normal_stl[0], -surface_normal_stl[1], -surface_normal_stl[2]])
# Camera should point INWARD (opposite of outward surface normal)
camera_forward = -normal_rotated

print(f"Camera Forward Direction: {camera_forward}")

# Calculate orientation
local_forward = np.array([0, 0, -1])  # Camera's -Z axis
rotation_axis = np.cross(local_forward, camera_forward)
rotation_axis_norm = np.linalg.norm(rotation_axis)

if rotation_axis_norm < 1e-6:
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

print(f"\nAxis-Angle Representation:")
print(f"  Axis: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
print(f"  Angle: {angle_rad:.6f} rad = {np.degrees(angle_rad):.1f}°")

# Verify
calculated_forward = rotation.apply(local_forward)
print(f"\nVerification:")
print(f"  Calculated forward: {calculated_forward}")
print(f"  Expected forward: {camera_forward}")
error_angle = np.degrees(np.arccos(np.clip(np.dot(calculated_forward, camera_forward), -1, 1)))
print(f"  Angular error: {error_angle:.6f}°")

print("\n" + "="*80)
print("FINAL CAMERA CONFIGURATION")
print("="*80)
print(f"\n<camera name=\"wrist_camera\"")
print(f"        pos=\"{camera_gripper[0]:.4f} {camera_gripper[1]:.4f} {camera_gripper[2]:.4f}\"")
print(f"        axisangle=\"{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f} {angle_rad:.6f}\"")
print(f"        fovy=\"140\"/>")

# Compare with previous configuration
print("\n" + "="*80)
print("COMPARISON WITH PREVIOUS CONFIGURATION")
print("="*80)

prev_pos = np.array([0.0025, 0.0742, 0.0046])
print(f"\nPrevious pos: [{prev_pos[0]:.4f}, {prev_pos[1]:.4f}, {prev_pos[2]:.4f}]")
print(f"New pos:      [{camera_gripper[0]:.4f}, {camera_gripper[1]:.4f}, {camera_gripper[2]:.4f}]")
print(f"Difference:   [{camera_gripper[0]-prev_pos[0]:.4f}, {camera_gripper[1]-prev_pos[1]:.4f}, {camera_gripper[2]-prev_pos[2]:.4f}]")
pos_change = np.linalg.norm(camera_gripper - prev_pos) * 1000
print(f"Position change: {pos_change:.2f} mm")

print("\n" + "="*80)
