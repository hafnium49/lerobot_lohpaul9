#!/usr/bin/env python3
"""
Analyze the dodecagon hole axis direction to determine correct camera orientation.
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
print("Analyzing Dodecagon Hole Axis Direction")
print("="*80)

# Read STL
stl_file = "src/lerobot/envs/so101_assets/official_model/assets/Wrist_Cam_Mount_32x32_UVC_Module_SO101_MIRRORED.stl"
print(f"\nReading: {stl_file}")
normals, vertices = read_stl_binary(stl_file)
vertices = vertices * 0.001

# Camera/hole center
hole_center = np.array([0.0025, -0.0611, -0.0111])

# Find hole wall triangles (near center, radial normals)
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

        if dist_xz < radius_threshold and abs(norm[1]) < 0.6:
            hole_wall_triangles.append(tri_verts)
            hole_wall_normals.append(norm)

hole_wall_triangles = np.array(hole_wall_triangles)
hole_wall_normals = np.array(hole_wall_normals)

print(f"\nFound {len(hole_wall_triangles)} hole wall triangles")

# Analyze the hole wall normals
print("\nAnalyzing Hole Wall Normals:")

# The hole wall normals should be radially pointing
# Calculate the average Y component
y_components = hole_wall_normals[:, 1]
print(f"  Y components: mean={y_components.mean():.4f}, std={y_components.std():.4f}")
print(f"  Y range: [{y_components.min():.4f}, {y_components.max():.4f}]")

# The hole axis is perpendicular to the radial normals
# For a cylindrical hole, the axis should be close to [0, ±1, 0]

# Method 1: Average the "perpendicular to radial" direction
# For each radial normal [nx, ny, nz], the hole axis has ny component
# But we can also analyze the hole geometry directly

# Method 2: Find the direction from back to front of the hole
hole_verts = hole_wall_triangles.reshape(-1, 3)
y_min_hole = hole_verts[:, 1].min()
y_max_hole = hole_verts[:, 1].max()

print(f"\nHole geometry:")
print(f"  Y range: [{y_min_hole:.4f}, {y_max_hole:.4f}] m")
print(f"  Y span: {(y_max_hole - y_min_hole)*1000:.1f} mm")

# Find vertices at front and back
front_verts = hole_verts[hole_verts[:, 1] > (y_max_hole - 0.003)]
back_verts = hole_verts[hole_verts[:, 1] < (y_min_hole + 0.003)]

print(f"  Front vertices: {len(front_verts)}")
print(f"  Back vertices: {len(back_verts)}")

if len(front_verts) > 0 and len(back_verts) > 0:
    front_center = front_verts.mean(axis=0)
    back_center = back_verts.mean(axis=0)

    print(f"\n  Front center: [{front_center[0]:.4f}, {front_center[1]:.4f}, {front_center[2]:.4f}]")
    print(f"  Back center:  [{back_center[0]:.4f}, {back_center[1]:.4f}, {back_center[2]:.4f}]")

    # Hole axis points from back to front
    hole_axis = front_center - back_center
    hole_axis = hole_axis / np.linalg.norm(hole_axis)

    print(f"\n  Hole axis (back->front): [{hole_axis[0]:.6f}, {hole_axis[1]:.6f}, {hole_axis[2]:.6f}]")

    # Angle from Y-axis
    y_axis = np.array([0, 1, 0])
    angle_from_y = np.degrees(np.arccos(np.clip(np.dot(hole_axis, y_axis), -1, 1)))
    print(f"  Angle from +Y axis: {angle_from_y:.1f}°")

# Method 3: Use cross products of radial normals
# The hole axis should be perpendicular to all radial normals
print("\nMethod 3: Calculating axis from radial normals...")

# For each pair of radial normals, their cross product gives the axis direction
axes = []
for i in range(min(20, len(hole_wall_normals))):
    for j in range(i+1, min(20, len(hole_wall_normals))):
        n1 = hole_wall_normals[i]
        n2 = hole_wall_normals[j]

        # Cross product
        axis = np.cross(n1, n2)
        axis_norm = np.linalg.norm(axis)

        if axis_norm > 0.1:  # Only if normals are not parallel
            axis = axis / axis_norm
            # Make sure pointing in +Y direction
            if axis[1] < 0:
                axis = -axis
            axes.append(axis)

if len(axes) > 0:
    axes = np.array(axes)
    mean_axis = axes.mean(axis=0)
    mean_axis = mean_axis / np.linalg.norm(mean_axis)

    print(f"  Mean axis from cross products: [{mean_axis[0]:.6f}, {mean_axis[1]:.6f}, {mean_axis[2]:.6f}]")
    angle_from_y = np.degrees(np.arccos(np.clip(np.dot(mean_axis, y_axis), -1, 1)))
    print(f"  Angle from +Y axis: {angle_from_y:.1f}°")

# Summary
print("\n" + "="*80)
print("HOLE AXIS DETERMINATION")
print("="*80)

# Use hole geometry method (most reliable)
hole_axis_stl = hole_axis  # From Method 2

print(f"\nHole Axis Direction (STL frame):")
print(f"  [{hole_axis_stl[0]:.6f}, {hole_axis_stl[1]:.6f}, {hole_axis_stl[2]:.6f}]")
print(f"  Angle from +Y: {angle_from_y:.1f}°")

# Camera should point along the hole axis (outward from mount)
# In STL frame, outward is +Y direction (toward fingertip)
camera_direction_stl = hole_axis_stl

print(f"\nCamera Direction (STL frame):")
print(f"  [{camera_direction_stl[0]:.6f}, {camera_direction_stl[1]:.6f}, {camera_direction_stl[2]:.6f}]")

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

# Camera position
camera_pos_gripper = np.array([0.0025, 0.0609, 0.0120])

print("\n" + "="*80)
print("CORRECTED CAMERA CONFIGURATION")
print("="*80)
print(f"\n<camera name=\"wrist_camera\"")
print(f"        pos=\"{camera_pos_gripper[0]:.4f} {camera_pos_gripper[1]:.4f} {camera_pos_gripper[2]:.4f}\"")
print(f"        axisangle=\"{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f} {angle_rad:.6f}\"")
print(f"        fovy=\"140\"/>")

# Compare with current
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

current_axis = np.array([1.0, 0.0, 0.0])
current_angle = 1.134477

print(f"\nCurrent:")
print(f"  Axis-angle: [{current_axis[0]:.6f}, {current_axis[1]:.6f}, {current_axis[2]:.6f}] {current_angle:.6f} rad")
print(f"  Angle: {np.degrees(current_angle):.1f}°")

print(f"\nCorrected (from hole axis):")
print(f"  Axis-angle: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}] {angle_rad:.6f} rad")
print(f"  Angle: {np.degrees(angle_rad):.1f}°")

angle_diff = abs(angle_rad - current_angle)
print(f"\nDifference:")
print(f"  Angle change: {np.degrees(angle_diff):.1f}°")

print("\n" + "="*80)
