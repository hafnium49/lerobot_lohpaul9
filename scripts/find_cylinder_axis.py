#!/usr/bin/env python3
"""
Find the true cylinder axis of the dodecagon lens hole.
The axis is perpendicular to all radial normals of the hole walls.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def read_stl_binary(filename):
    """Read binary STL file."""
    with open(filename, 'rb') as f:
        f.read(80)
        num_triangles = np.frombuffer(f.read(4), dtype=np.uint32)[0]

        vertices = []
        normals = []

        for _ in range(num_triangles):
            normal = np.frombuffer(f.read(12), dtype=np.float32)
            normals.append(normal)

            triangle_vertices = []
            for _ in range(3):
                vertex = np.frombuffer(f.read(12), dtype=np.float32)
                triangle_vertices.append(vertex)
            vertices.append(triangle_vertices)

            f.read(2)

        return np.array(normals), np.array(vertices)


print("="*80)
print("Finding Cylinder Axis (Perpendicular to All Hole Wall Normals)")
print("="*80)

# Read STL
stl_file = "src/lerobot/envs/so101_assets/official_model/assets/Wrist_Cam_Mount_32x32_UVC_Module_SO101_MIRRORED.stl"
print(f"\nReading: {stl_file}")
normals, vertices = read_stl_binary(stl_file)
vertices = vertices * 0.001

# Camera/hole center
hole_center = np.array([0.0025, -0.0611, -0.0111])

# Find hole wall triangles (near center, radial normals with small Y component)
tip_threshold = -0.0594
radius_threshold = 0.012

hole_wall_triangles = []
hole_wall_normals = []

for tri_verts, normal in zip(vertices, normals):
    tri_y_min = tri_verts[:, 1].min()
    if tri_y_min < tip_threshold:
        norm = normal / np.linalg.norm(normal)

        tri_center = tri_verts.mean(axis=0)
        dx = tri_center[0] - hole_center[0]
        dz = tri_center[2] - hole_center[2]
        dist_xz = np.sqrt(dx**2 + dz**2)

        # Hole walls have small Y component (radial normals)
        if dist_xz < radius_threshold and abs(norm[1]) < 0.5:
            hole_wall_triangles.append(tri_verts)
            hole_wall_normals.append(norm)

hole_wall_triangles = np.array(hole_wall_triangles)
hole_wall_normals = np.array(hole_wall_normals)

print(f"\nFound {len(hole_wall_triangles)} hole wall triangles with radial normals")

print("\nAnalyzing radial normals:")
print(f"  Y component range: [{hole_wall_normals[:, 1].min():.3f}, {hole_wall_normals[:, 1].max():.3f}]")
print(f"  Y component mean: {hole_wall_normals[:, 1].mean():.3f}")

# Method: The cylinder axis is perpendicular to all radial normals
# For a perfect cylinder with axis A, all radial normals N satisfy: N · A = 0
# We find A by solving the least-squares problem

# The axis should be perpendicular to each radial normal
# Set up system: for each normal n, we want axis · n = 0
# This means axis is in the null space of the matrix of normals

print("\nMethod: Finding axis perpendicular to all radial normals...")

# Use SVD to find the null space (the axis direction)
U, S, Vt = np.linalg.svd(hole_wall_normals)

# The smallest singular value corresponds to the cylinder axis
print(f"\nSingular values: {S[:5]}")

# The axis is the right singular vector corresponding to smallest singular value
axis_candidate = Vt[-1, :]  # Last row of Vt (corresponding to smallest singular value)

print(f"\nCandidate axis (from SVD null space):")
print(f"  [{axis_candidate[0]:.6f}, {axis_candidate[1]:.6f}, {axis_candidate[2]:.6f}]")

# Make sure it points in +Y direction (outward from mount)
if axis_candidate[1] < 0:
    axis_candidate = -axis_candidate

print(f"  (oriented +Y): [{axis_candidate[0]:.6f}, {axis_candidate[1]:.6f}, {axis_candidate[2]:.6f}]")

# Verify: check that this axis is perpendicular to radial normals
dot_products = np.abs(hole_wall_normals @ axis_candidate)
print(f"\nVerification (axis · normals):")
print(f"  Mean: {dot_products.mean():.6f}")
print(f"  Max:  {dot_products.max():.6f}")
print(f"  Std:  {dot_products.std():.6f}")

if dot_products.mean() < 0.1:
    print(f"  ✅ Axis is approximately perpendicular to all normals!")
else:
    print(f"  ⚠️  Axis may not be perfectly perpendicular")

# Angle from Y-axis
y_axis = np.array([0, 1, 0])
angle_from_y = np.degrees(np.arccos(np.clip(np.dot(axis_candidate, y_axis), -1, 1)))
print(f"\nAngle from +Y axis: {angle_from_y:.2f}°")

# This is the camera direction in STL frame
camera_direction_stl = axis_candidate

print("\n" + "="*80)
print("CAMERA CONFIGURATION")
print("="*80)

print(f"\nCamera Direction (STL frame):")
print(f"  [{camera_direction_stl[0]:.6f}, {camera_direction_stl[1]:.6f}, {camera_direction_stl[2]:.6f}]")
print(f"  (parallel to cylinder axis, perpendicular to all hole wall normals)")

# Transform to gripper frame
# 180° rotation around Y: (x, y, z) -> (x, -y, -z)
camera_direction_gripper = np.array([
    camera_direction_stl[0],
    -camera_direction_stl[1],
    -camera_direction_stl[2]
])

print(f"\nCamera Direction (Gripper frame):")
print(f"  [{camera_direction_gripper[0]:.6f}, {camera_direction_gripper[1]:.6f}, {camera_direction_gripper[2]:.6f}]")

# Calculate orientation
local_forward = np.array([0, 0, -1])  # Camera -Z axis

rotation_axis = np.cross(local_forward, camera_direction_gripper)
rotation_axis_norm = np.linalg.norm(rotation_axis)

if rotation_axis_norm < 1e-6:
    if np.dot(local_forward, camera_direction_gripper) > 0:
        rotation = R.from_euler('xyz', [0, 0, 0])
    else:
        rotation = R.from_euler('xyz', [np.pi, 0, 0])
else:
    rotation_axis = rotation_axis / rotation_axis_norm
    angle = np.arccos(np.clip(np.dot(local_forward, camera_direction_gripper), -1, 1))
    rotation = R.from_rotvec(rotation_axis * angle)

# Axis-angle
rotvec = rotation.as_rotvec()
angle_rad = np.linalg.norm(rotvec)
if angle_rad > 1e-6:
    axis = rotvec / angle_rad
else:
    axis = np.array([1, 0, 0])
    angle_rad = 0

print(f"\nCamera Orientation (Axis-Angle):")
print(f"  Axis:  [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
print(f"  Angle: {angle_rad:.6f} rad = {np.degrees(angle_rad):.1f}°")

# Verify
calculated_forward = rotation.apply(local_forward)
angular_error = np.degrees(np.arccos(np.clip(np.dot(calculated_forward, camera_direction_gripper), -1, 1)))

print(f"\nVerification:")
print(f"  Calculated forward: [{calculated_forward[0]:.6f}, {calculated_forward[1]:.6f}, {calculated_forward[2]:.6f}]")
print(f"  Expected forward:   [{camera_direction_gripper[0]:.6f}, {camera_direction_gripper[1]:.6f}, {camera_direction_gripper[2]:.6f}]")
print(f"  Angular error: {angular_error:.6f}°")

# Camera position
camera_pos_gripper = np.array([0.0025, 0.0609, 0.0120])

print("\n" + "="*80)
print("FINAL CAMERA CONFIGURATION")
print("="*80)
print(f"\n<camera name=\"wrist_camera\"")
print(f"        pos=\"{camera_pos_gripper[0]:.4f} {camera_pos_gripper[1]:.4f} {camera_pos_gripper[2]:.4f}\"")
print(f"        axisangle=\"{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f} {angle_rad:.6f}\"")
print(f"        fovy=\"140\"/>")

# Compare with current
print("\n" + "="*80)
print("COMPARISON WITH CURRENT")
print("="*80)

current_axis = np.array([-0.997248, 0.074142, 0.000000])
current_angle = 1.953575

print(f"\nCurrent:")
print(f"  Axis-angle: [{current_axis[0]:.6f}, {current_axis[1]:.6f}, {current_axis[2]:.6f}] {current_angle:.6f} rad")
print(f"  Angle: {np.degrees(current_angle):.1f}°")

print(f"\nCorrected (perpendicular to all hole walls):")
print(f"  Axis-angle: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}] {angle_rad:.6f} rad")
print(f"  Angle: {np.degrees(angle_rad):.1f}°")

print("\n" + "="*80)
