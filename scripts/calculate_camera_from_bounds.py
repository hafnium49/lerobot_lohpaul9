#!/usr/bin/env python3
"""
Calculate camera position based on the bounding box of mounting surface.
Since the STL only models the rim/edges, we'll use the bounds to estimate the full surface.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


print("="*80)
print("Camera Position from Mounting Surface Bounds")
print("="*80)

# From the visualization, we found:
# - Surface at tip with normal [0, 0.9063, -0.4226] (25° tilt)
# - X range: [-0.0110, 0.0160] m (27mm width)
# - Y at surface: -0.0744 m
# - Z range: [-0.0056, -0.0018] m (only 3.8mm - incomplete geometry)

# The X dimension (~27-28mm) is close to 32mm, so we'll trust it
# For Z, we need to estimate the full height

# Strategy: Assume symmetric square mounting surface
# Use X bounds but extend Z bounds to make it square

x_min, x_max = -0.0110, 0.0160  # 27mm - trust this
y_surface = -0.0744  # Surface Y position
z_center_observed = (-0.0056 + -0.0018) / 2  # Center of observed Z strip

# The observed Z range is only the top/bottom edges
# For a 32mm square, if X is 27mm, scale Z proportionally
# But we know it should be ~32mm square, so let's use that
surface_size = 0.032  # 32mm

# Center should be at middle of X range
center_x = (x_min + x_max) / 2

# For Z, we have incomplete geometry
# Assuming the camera lens hole is at the center, use observed Z center
# But expand to account for full surface
center_z = z_center_observed  # Use observed center

# But we can also estimate from assuming square:
x_width = x_max - x_min  # 27mm
# If it's meant to be 32mm, extend X slightly and make Z symmetric
center_x_symmetric = (x_min + x_max) / 2
z_half_size = x_width / 2  # Assume square
center_z_symmetric = z_center_observed  # Keep at observed center

print(f"\nObserved mounting surface:")
print(f"  X: [{x_min:.4f}, {x_max:.4f}] m (width: {(x_max-x_min)*1000:.1f} mm)")
print(f"  Y: {y_surface:.4f} m")
print(f"  Z: observed center = {z_center_observed:.4f} m")

print(f"\nEstimated center (dodecagon lens hole) in STL frame:")
print(f"  X: {center_x:.4f} m")
print(f"  Y: {y_surface:.4f} m")
print(f"  Z: {center_z:.4f} m")

# Surface normal (inward toward workspace after coordinate transform)
surface_normal_stl = np.array([0.0, 0.9063, -0.4226])
surface_normal_stl = surface_normal_stl / np.linalg.norm(surface_normal_stl)

print(f"\nSurface normal (STL): {surface_normal_stl}")

# Transform to gripper frame
# Camera mount: pos=[0, -0.000218214, 0.000949706], quat=[0, 1, 0, 0]
# quat [0,1,0,0] = 180° rotation around Y-axis
# Transformation: (x,y,z) -> (x, -y, -z) + mount_pos

mount_pos = np.array([0.0, -0.000218214, 0.000949706])
camera_stl = np.array([center_x, y_surface, center_z])
camera_rotated = np.array([camera_stl[0], -camera_stl[1], -camera_stl[2]])
camera_gripper = mount_pos + camera_rotated

print(f"\nCamera position in gripper frame:")
print(f"  [{camera_gripper[0]:.4f}, {camera_gripper[1]:.4f}, {camera_gripper[2]:.4f}]")

# Surface normal after rotation: (x, y, z) -> (x, -y, -z)
normal_rotated = np.array([surface_normal_stl[0], -surface_normal_stl[1], -surface_normal_stl[2]])
# Camera should point INWARD (opposite of outward surface normal)
camera_forward = -normal_rotated

print(f"\nCamera should point toward: {camera_forward}")

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

print(f"\nAxis-angle representation:")
print(f"  Axis: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
print(f"  Angle: {angle_rad:.6f} rad = {np.degrees(angle_rad):.1f}°")

# Verify
calculated_forward = rotation.apply(local_forward)
print(f"\nVerification:")
print(f"  Calculated forward: {calculated_forward}")
print(f"  Expected forward: {camera_forward}")
error = np.linalg.norm(calculated_forward - camera_forward)
print(f"  Error: {error:.9f}")

print("\n" + "="*80)
print("FINAL CAMERA CONFIGURATION")
print("="*80)
print(f"\n<camera name=\"wrist_camera\"")
print(f"        pos=\"{camera_gripper[0]:.4f} {camera_gripper[1]:.4f} {camera_gripper[2]:.4f}\"")
print(f"        axisangle=\"{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f} {angle_rad:.6f}\"")
print(f"        fovy=\"140\"/>")
print("\n" + "="*80)
