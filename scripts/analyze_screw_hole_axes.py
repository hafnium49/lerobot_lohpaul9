#!/usr/bin/env python3
"""
Analyze the 4 screw holes to find their cylindrical axes.
All 4 holes should have parallel axes perpendicular to the mounting surface.
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
print("Analyzing 4 Screw Hole Cylindrical Axes")
print("="*80)

# Read STL
stl_file = "src/lerobot/envs/so101_assets/official_model/assets/Wrist_Cam_Mount_32x32_UVC_Module_SO101_MIRRORED.stl"
print(f"\nReading: {stl_file}")
normals, vertices = read_stl_binary(stl_file)
vertices = vertices * 0.001

# The 4 corner screw holes (from previous analysis)
screw_holes_stl = np.array([
    [-0.0124, -0.0611, -0.0107],  # H1 (bottom-left)
    [-0.0131, -0.0858,  0.0019],  # H4 (top-left)
    [ 0.0172, -0.0616, -0.0110],  # H6 (bottom-right)
    [ 0.0181, -0.0859,  0.0019],  # H9 (top-right)
])

labels = ["H1 (bottom-left)", "H4 (top-left)", "H6 (bottom-right)", "H9 (top-right)"]

print("\n4 Screw Hole Centers (STL frame):")
for i, (hole, label) in enumerate(zip(screw_holes_stl, labels)):
    print(f"  {label}: [{hole[0]:.4f}, {hole[1]:.4f}, {hole[2]:.4f}] m")

# Find triangles around each screw hole
all_hole_axes = []
all_hole_triangles = []

tip_threshold = -0.0594  # At tip

for hole_idx, (hole_center, label) in enumerate(zip(screw_holes_stl, labels)):
    print(f"\n{'='*80}")
    print(f"Analyzing {label}")
    print(f"{'='*80}")

    # Find triangles near this hole (within 8mm radius in XZ)
    radius_threshold = 0.008

    hole_triangles = []
    hole_normals = []

    for tri_verts, normal in zip(vertices, normals):
        tri_y_min = tri_verts[:, 1].min()
        if tri_y_min < tip_threshold:
            norm = normal / np.linalg.norm(normal)

            tri_center = tri_verts.mean(axis=0)
            dx = tri_center[0] - hole_center[0]
            dz = tri_center[2] - hole_center[2]
            dist_xz = np.sqrt(dx**2 + dz**2)

            # Hole walls have radial normals (small Y component)
            if dist_xz < radius_threshold and abs(norm[1]) < 0.5:
                hole_triangles.append(tri_verts)
                hole_normals.append(norm)

    if len(hole_triangles) == 0:
        print(f"  ⚠️  No hole wall triangles found")
        continue

    hole_triangles = np.array(hole_triangles)
    hole_normals = np.array(hole_normals)

    print(f"  Found {len(hole_triangles)} hole wall triangles")
    print(f"  Y component range: [{hole_normals[:, 1].min():.3f}, {hole_normals[:, 1].max():.3f}]")

    # Find cylinder axis using SVD (perpendicular to all radial normals)
    U, S, Vt = np.linalg.svd(hole_normals)

    # The axis is the right singular vector corresponding to smallest singular value
    axis_candidate = Vt[-1, :]

    # Make sure it points in +Y direction (outward from mount)
    if axis_candidate[1] < 0:
        axis_candidate = -axis_candidate

    print(f"  Cylinder axis: [{axis_candidate[0]:.6f}, {axis_candidate[1]:.6f}, {axis_candidate[2]:.6f}]")

    # Verify perpendicularity
    dot_products = np.abs(hole_normals @ axis_candidate)
    print(f"  Verification (axis · normals):")
    print(f"    Mean: {dot_products.mean():.6f}")
    print(f"    Max:  {dot_products.max():.6f}")

    # Angle from Y-axis
    y_axis = np.array([0, 1, 0])
    angle_from_y = np.degrees(np.arccos(np.clip(np.dot(axis_candidate, y_axis), -1, 1)))
    print(f"  Angle from +Y axis: {angle_from_y:.2f}°")

    all_hole_axes.append(axis_candidate)
    all_hole_triangles.append(hole_triangles)

# Analyze consistency across all holes
print(f"\n{'='*80}")
print("CONSISTENCY ANALYSIS")
print(f"{'='*80}")

if len(all_hole_axes) > 0:
    all_hole_axes = np.array(all_hole_axes)

    print(f"\nFound axes for {len(all_hole_axes)} holes")
    print("\nAll hole axes (STL frame):")
    for i, (axis, label) in enumerate(zip(all_hole_axes, labels)):
        angle_from_y = np.degrees(np.arccos(np.clip(np.dot(axis, y_axis), -1, 1)))
        print(f"  {label}: [{axis[0]:7.4f}, {axis[1]:7.4f}, {axis[2]:7.4f}]  (angle from Y: {angle_from_y:.1f}°)")

    # Calculate mean axis
    mean_axis = all_hole_axes.mean(axis=0)
    mean_axis = mean_axis / np.linalg.norm(mean_axis)

    print(f"\nMean axis: [{mean_axis[0]:.6f}, {mean_axis[1]:.6f}, {mean_axis[2]:.6f}]")

    # Calculate standard deviation of axes
    axis_deviations = []
    for axis in all_hole_axes:
        angle_diff = np.degrees(np.arccos(np.clip(np.dot(axis, mean_axis), -1, 1)))
        axis_deviations.append(angle_diff)

    print(f"  Angular deviation from mean:")
    print(f"    Mean: {np.mean(axis_deviations):.2f}°")
    print(f"    Max:  {np.max(axis_deviations):.2f}°")
    print(f"    Std:  {np.std(axis_deviations):.2f}°")

    if np.max(axis_deviations) < 5.0:
        print(f"  ✅ All holes have consistent axes (variation < 5°)")
    else:
        print(f"  ⚠️  Holes have inconsistent axes (variation > 5°)")

    # Mean angle from Y
    mean_angle_from_y = np.degrees(np.arccos(np.clip(np.dot(mean_axis, y_axis), -1, 1)))
    print(f"\nMean axis angle from +Y: {mean_angle_from_y:.2f}°")

    # This mean axis is the mounting surface normal
    mounting_normal_stl = mean_axis

    print(f"\n{'='*80}")
    print("MOUNTING SURFACE NORMAL (from screw hole axes)")
    print(f"{'='*80}")
    print(f"\nSTL frame: [{mounting_normal_stl[0]:.6f}, {mounting_normal_stl[1]:.6f}, {mounting_normal_stl[2]:.6f}]")
    print(f"Angle from +Y axis: {mean_angle_from_y:.2f}°")

    # Camera should point along this normal (outward through holes)
    camera_direction_stl = mounting_normal_stl

    print(f"\n{'='*80}")
    print("CAMERA ORIENTATION")
    print(f"{'='*80}")

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

    print(f"\n{'='*80}")
    print("CAMERA CONFIGURATION FROM SCREW HOLE AXES")
    print(f"{'='*80}")
    print(f"\n<camera name=\"wrist_camera\"")
    print(f"        pos=\"{camera_pos_gripper[0]:.4f} {camera_pos_gripper[1]:.4f} {camera_pos_gripper[2]:.4f}\"")
    print(f"        axisangle=\"{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f} {angle_rad:.6f}\"")
    print(f"        fovy=\"140\"/>")

    # Compare with current
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")

    current_axis = np.array([-0.999980, 0.006384, 0.000000])
    current_angle = 2.007098

    print(f"\nCurrent (from dodecagon hole):")
    print(f"  Axis-angle: [{current_axis[0]:.6f}, {current_axis[1]:.6f}, {current_axis[2]:.6f}] {current_angle:.6f} rad")
    print(f"  Angle: {np.degrees(current_angle):.1f}°")

    print(f"\nCorrected (from screw hole axes):")
    print(f"  Axis-angle: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}] {angle_rad:.6f} rad")
    print(f"  Angle: {np.degrees(angle_rad):.1f}°")

    angle_diff = abs(angle_rad - current_angle)
    print(f"\nDifference:")
    print(f"  Angle change: {np.degrees(angle_diff):.1f}°")

print(f"\n{'='*80}")
