#!/usr/bin/env python3
"""
Calculate camera position and orientation from the dodecagon lens hole geometry.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


print("="*80)
print("Camera Configuration from Dodecagon Lens Hole")
print("="*80)

# From dodecagon extraction results
print("\nDodecagon Lens Hole Geometry:")
print("  Hole center (XZ): [2.5, -11.1] mm")
print("  Hole center (X): 2.5 mm = 0.0025 m")
print("  Hole center (Z): -11.1 mm = -0.0111 m")
print("  Surface Y: -61.1 mm = -0.0611 m")
print("  Hole diameter: ~28.0 mm")

# More accurate center position in STL frame
# Use the hole center from dodecagon extraction
dodecagon_center_stl = np.array([0.0025, -0.0611, -0.0111])

print(f"\nDodecagon Lens Hole Center (STL frame):")
print(f"  [{dodecagon_center_stl[0]:.4f}, {dodecagon_center_stl[1]:.4f}, {dodecagon_center_stl[2]:.4f}] m")
print(f"  [{dodecagon_center_stl[0]*1000:.1f}, {dodecagon_center_stl[1]*1000:.1f}, {dodecagon_center_stl[2]*1000:.1f}] mm")

# Compare with 4-screw-hole center
screw_hole_center_stl = np.array([0.0024, -0.0611, -0.0045])
difference = dodecagon_center_stl - screw_hole_center_stl

print(f"\nComparison with 4-Screw-Hole Center:")
print(f"  Screw holes: [{screw_hole_center_stl[0]:.4f}, {screw_hole_center_stl[1]:.4f}, {screw_hole_center_stl[2]:.4f}] m")
print(f"  Dodecagon:   [{dodecagon_center_stl[0]:.4f}, {dodecagon_center_stl[1]:.4f}, {dodecagon_center_stl[2]:.4f}] m")
print(f"  Difference:  [{difference[0]*1000:.1f}, {difference[1]*1000:.1f}, {difference[2]*1000:.1f}] mm")

offset_magnitude = np.linalg.norm(difference) * 1000
print(f"  Offset magnitude: {offset_magnitude:.1f} mm")

# Surface normal from dodecagon hole analysis
# The hole walls have radially-pointing normals
# The surface normal (perpendicular to hole) is the mounting surface normal
surface_normal_stl = np.array([0.0, 0.9063, -0.4226])
surface_normal_stl = surface_normal_stl / np.linalg.norm(surface_normal_stl)

print(f"\nSurface Normal (perpendicular to dodecagon hole):")
print(f"  STL frame: [{surface_normal_stl[0]:.6f}, {surface_normal_stl[1]:.6f}, {surface_normal_stl[2]:.6f}]")

tilt_angle = np.degrees(np.arccos(abs(surface_normal_stl[1])))
print(f"  Tilt from Y-axis: {tilt_angle:.1f}°")

# Transform to gripper frame
# Camera mount: pos=[0, -0.000218214, 0.000949706], quat=[0, 1, 0, 0]
# quat [0,1,0,0] = 180° rotation around Y-axis
# Transformation: (x,y,z) -> (x, -y, -z) + mount_pos

mount_pos = np.array([0.0, -0.000218214, 0.000949706])

print(f"\nCamera Mount Transformation:")
print(f"  Position: [{mount_pos[0]:.9f}, {mount_pos[1]:.9f}, {mount_pos[2]:.9f}] m")
print(f"  Rotation: quat=[0, 1, 0, 0] (180° around Y-axis)")
print(f"  Formula: (x, y, z) -> (x, -y, -z) + mount_pos")

# Apply transformation to dodecagon center
camera_rotated = np.array([dodecagon_center_stl[0], -dodecagon_center_stl[1], -dodecagon_center_stl[2]])
camera_gripper = mount_pos + camera_rotated

print(f"\nCamera Position (Gripper frame):")
print(f"  [{camera_gripper[0]:.4f}, {camera_gripper[1]:.4f}, {camera_gripper[2]:.4f}] m")
print(f"  [{camera_gripper[0]*1000:.1f}, {camera_gripper[1]*1000:.1f}, {camera_gripper[2]*1000:.1f}] mm")

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

# Compare with current configuration
print("\n" + "="*80)
print("COMPARISON WITH CURRENT CONFIGURATION")
print("="*80)

current_pos = np.array([0.0024, 0.0609, 0.0054])
current_axis = np.array([1.0, 0.0, 0.0])
current_angle = 1.134477

print(f"\nCurrent (from 4 screw holes):")
print(f"  Position: [{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}] m")
print(f"  Axis-angle: [{current_axis[0]:.6f}, {current_axis[1]:.6f}, {current_axis[2]:.6f}] {current_angle:.6f} rad")

print(f"\nNew (from dodecagon lens hole):")
print(f"  Position: [{camera_gripper[0]:.4f}, {camera_gripper[1]:.4f}, {camera_gripper[2]:.4f}] m")
print(f"  Axis-angle: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}] {angle_rad:.6f} rad")

pos_diff = camera_gripper - current_pos
angle_diff = abs(angle_rad - current_angle)

print(f"\nDifference:")
print(f"  Position: [{pos_diff[0]*1000:.1f}, {pos_diff[1]*1000:.1f}, {pos_diff[2]*1000:.1f}] mm")
print(f"  Position magnitude: {np.linalg.norm(pos_diff)*1000:.1f} mm")
print(f"  Angle: {np.degrees(angle_diff):.3f}°")

# Final configuration
print("\n" + "="*80)
print("FINAL CAMERA CONFIGURATION (from Dodecagon Lens Hole)")
print("="*80)
print(f"\n<camera name=\"wrist_camera\"")
print(f"        pos=\"{camera_gripper[0]:.4f} {camera_gripper[1]:.4f} {camera_gripper[2]:.4f}\"")
print(f"        axisangle=\"{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f} {angle_rad:.6f}\"")
print(f"        fovy=\"140\"/>")

if np.linalg.norm(pos_diff)*1000 > 1.0:
    print(f"\n⚠️  Position difference is {np.linalg.norm(pos_diff)*1000:.1f} mm")
    print(f"   Dodecagon center differs from 4-screw-hole center")
    print(f"   Recommend updating XML configuration")
else:
    print(f"\n✅ Position difference is small ({np.linalg.norm(pos_diff)*1000:.1f} mm)")
    print(f"   Current configuration is acceptable")

print("\n" + "="*80)
