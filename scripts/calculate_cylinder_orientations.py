#!/usr/bin/env python3
"""
Calculate complete spatial orientations of the 4 screw hole cylinder axes.
Provides multiple angular representations: polar coordinates, plane angles, and Euler angles.
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


def calculate_polar_angles(axis):
    """
    Calculate polar coordinates (spherical) for a 3D vector.

    Returns:
        theta: angle from +Z axis (elevation) in degrees [0, 180]
        phi: angle in XY plane from +X axis (azimuth) in degrees [0, 360]
    """
    x, y, z = axis

    # Theta: angle from +Z axis
    theta = np.degrees(np.arccos(np.clip(z / np.linalg.norm(axis), -1, 1)))

    # Phi: angle in XY plane from +X axis
    phi = np.degrees(np.arctan2(y, x))
    if phi < 0:
        phi += 360

    return theta, phi


def calculate_plane_angles(axis):
    """
    Calculate angles in each coordinate plane.

    Returns:
        xy_angle: angle from +X in XY plane (degrees)
        xz_angle: angle from +X in XZ plane (degrees)
        yz_angle: angle from +Y in YZ plane (degrees)
    """
    x, y, z = axis

    # XY plane: angle from +X axis
    xy_angle = np.degrees(np.arctan2(y, x))

    # XZ plane: angle from +X axis
    xz_angle = np.degrees(np.arctan2(z, x))

    # YZ plane: angle from +Y axis
    yz_angle = np.degrees(np.arctan2(z, y))

    return xy_angle, xz_angle, yz_angle


def calculate_euler_angles(axis, reference_axis=np.array([0, 1, 0])):
    """
    Calculate Euler angles (XYZ convention) to rotate from reference axis to target axis.

    Returns:
        roll, pitch, yaw in degrees
    """
    # Calculate rotation from reference to target
    rotation_axis = np.cross(reference_axis, axis)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    if rotation_axis_norm < 1e-6:
        # Parallel or anti-parallel
        if np.dot(reference_axis, axis) > 0:
            rotation = R.from_euler('xyz', [0, 0, 0])
        else:
            rotation = R.from_euler('xyz', [np.pi, 0, 0])
    else:
        rotation_axis = rotation_axis / rotation_axis_norm
        angle = np.arccos(np.clip(np.dot(reference_axis, axis), -1, 1))
        rotation = R.from_rotvec(rotation_axis * angle)

    # Get Euler angles
    euler = rotation.as_euler('xyz', degrees=True)

    return euler[0], euler[1], euler[2]


print("="*80)
print("Complete Spatial Orientations of 4 Screw Hole Cylinder Axes")
print("="*80)

# Read STL
stl_file = "src/lerobot/envs/so101_assets/official_model/assets/Wrist_Cam_Mount_32x32_UVC_Module_SO101_MIRRORED.stl"
print(f"\nReading: {stl_file}")
normals, vertices = read_stl_binary(stl_file)
vertices = vertices * 0.001

# The 4 corner screw holes
screw_holes_stl = np.array([
    [-0.0124, -0.0611, -0.0107],  # H1 (bottom-left)
    [-0.0131, -0.0858,  0.0019],  # H4 (top-left)
    [ 0.0172, -0.0616, -0.0110],  # H6 (bottom-right)
    [ 0.0181, -0.0859,  0.0019],  # H9 (top-right)
])

labels = ["H1 (bottom-left)", "H4 (top-left)", "H6 (bottom-right)", "H9 (top-right)"]

# Find cylinder axes for each hole
all_hole_axes = []
tip_threshold = -0.0594

for hole_idx, (hole_center, label) in enumerate(zip(screw_holes_stl, labels)):
    radius_threshold = 0.008

    hole_normals = []

    for tri_verts, normal in zip(vertices, normals):
        tri_y_min = tri_verts[:, 1].min()
        if tri_y_min < tip_threshold:
            norm = normal / np.linalg.norm(normal)

            tri_center = tri_verts.mean(axis=0)
            dx = tri_center[0] - hole_center[0]
            dz = tri_center[2] - hole_center[2]
            dist_xz = np.sqrt(dx**2 + dz**2)

            if dist_xz < radius_threshold and abs(norm[1]) < 0.5:
                hole_normals.append(norm)

    if len(hole_normals) == 0:
        continue

    hole_normals = np.array(hole_normals)

    # Find cylinder axis using SVD
    U, S, Vt = np.linalg.svd(hole_normals)
    axis_candidate = Vt[-1, :]

    # Orient in +Y direction
    if axis_candidate[1] < 0:
        axis_candidate = -axis_candidate

    all_hole_axes.append(axis_candidate)

all_hole_axes = np.array(all_hole_axes)

print(f"\n{'='*80}")
print("CYLINDER AXES (STL Frame)")
print(f"{'='*80}")

for i, (axis, label) in enumerate(zip(all_hole_axes, labels)):
    print(f"\n{label}:")
    print(f"  Axis vector: [{axis[0]:7.4f}, {axis[1]:7.4f}, {axis[2]:7.4f}]")

# Calculate mean axis
mean_axis = all_hole_axes.mean(axis=0)
mean_axis = mean_axis / np.linalg.norm(mean_axis)

print(f"\n{'='*80}")
print("SPATIAL ORIENTATIONS (Multiple Representations)")
print(f"{'='*80}")

# Reference axes
y_axis = np.array([0, 1, 0])
x_axis = np.array([1, 0, 0])
z_axis = np.array([0, 0, 1])

print("\n" + "-"*80)
print("Individual Hole Orientations:")
print("-"*80)

for i, (axis, label) in enumerate(zip(all_hole_axes, labels)):
    print(f"\n{label}:")
    print(f"  Axis: [{axis[0]:7.4f}, {axis[1]:7.4f}, {axis[2]:7.4f}]")

    # Angles from coordinate axes
    angle_from_x = np.degrees(np.arccos(np.clip(np.dot(axis, x_axis), -1, 1)))
    angle_from_y = np.degrees(np.arccos(np.clip(np.dot(axis, y_axis), -1, 1)))
    angle_from_z = np.degrees(np.arccos(np.clip(np.dot(axis, z_axis), -1, 1)))

    print(f"  Angles from coordinate axes:")
    print(f"    From +X: {angle_from_x:6.2f}°")
    print(f"    From +Y: {angle_from_y:6.2f}°")
    print(f"    From +Z: {angle_from_z:6.2f}°")

    # Polar coordinates (spherical)
    theta, phi = calculate_polar_angles(axis)
    print(f"  Polar coordinates (spherical):")
    print(f"    θ (from +Z): {theta:6.2f}°")
    print(f"    φ (azimuth):  {phi:6.2f}°")

    # Plane angles
    xy_angle, xz_angle, yz_angle = calculate_plane_angles(axis)
    print(f"  Coordinate plane angles:")
    print(f"    XY plane (from +X): {xy_angle:6.2f}°")
    print(f"    XZ plane (from +X): {xz_angle:6.2f}°")
    print(f"    YZ plane (from +Y): {yz_angle:6.2f}°")

    # Euler angles (rotation from +Y to axis)
    roll, pitch, yaw = calculate_euler_angles(axis, y_axis)
    print(f"  Euler angles (XYZ, from +Y):")
    print(f"    Roll:  {roll:6.2f}°")
    print(f"    Pitch: {pitch:6.2f}°")
    print(f"    Yaw:   {yaw:6.2f}°")

print(f"\n{'='*80}")
print("MEAN AXIS ORIENTATION")
print(f"{'='*80}")

print(f"\nMean axis: [{mean_axis[0]:7.4f}, {mean_axis[1]:7.4f}, {mean_axis[2]:7.4f}]")

# Angles from coordinate axes
angle_from_x = np.degrees(np.arccos(np.clip(np.dot(mean_axis, x_axis), -1, 1)))
angle_from_y = np.degrees(np.arccos(np.clip(np.dot(mean_axis, y_axis), -1, 1)))
angle_from_z = np.degrees(np.arccos(np.clip(np.dot(mean_axis, z_axis), -1, 1)))

print(f"\nAngles from coordinate axes:")
print(f"  From +X: {angle_from_x:6.2f}°")
print(f"  From +Y: {angle_from_y:6.2f}°")
print(f"  From +Z: {angle_from_z:6.2f}°")

# Polar coordinates
theta, phi = calculate_polar_angles(mean_axis)
print(f"\nPolar coordinates (spherical):")
print(f"  θ (elevation from +Z): {theta:6.2f}°")
print(f"  φ (azimuth in XY):     {phi:6.2f}°")

# Plane angles
xy_angle, xz_angle, yz_angle = calculate_plane_angles(mean_axis)
print(f"\nCoordinate plane angles:")
print(f"  XY plane (from +X): {xy_angle:6.2f}°")
print(f"  XZ plane (from +X): {xz_angle:6.2f}°")
print(f"  YZ plane (from +Y): {yz_angle:6.2f}°")

# Euler angles
roll, pitch, yaw = calculate_euler_angles(mean_axis, y_axis)
print(f"\nEuler angles (XYZ convention, rotation from +Y):")
print(f"  Roll  (X): {roll:6.2f}°")
print(f"  Pitch (Y): {pitch:6.2f}°")
print(f"  Yaw   (Z): {yaw:6.2f}°")

# Transform to gripper frame (180° rotation around Y)
mean_axis_gripper = np.array([mean_axis[0], -mean_axis[1], -mean_axis[2]])

print(f"\n{'='*80}")
print("MEAN AXIS IN GRIPPER FRAME")
print(f"{'='*80}")

print(f"\nAxis (gripper frame): [{mean_axis_gripper[0]:7.4f}, {mean_axis_gripper[1]:7.4f}, {mean_axis_gripper[2]:7.4f}]")

# Angles from coordinate axes (gripper frame)
angle_from_x_g = np.degrees(np.arccos(np.clip(np.dot(mean_axis_gripper, x_axis), -1, 1)))
angle_from_y_g = np.degrees(np.arccos(np.clip(np.dot(mean_axis_gripper, y_axis), -1, 1)))
angle_from_z_g = np.degrees(np.arccos(np.clip(np.dot(mean_axis_gripper, z_axis), -1, 1)))

print(f"\nAngles from coordinate axes (gripper frame):")
print(f"  From +X: {angle_from_x_g:6.2f}°")
print(f"  From +Y: {angle_from_y_g:6.2f}°")
print(f"  From +Z: {angle_from_z_g:6.2f}°")

# Polar coordinates (gripper frame)
theta_g, phi_g = calculate_polar_angles(mean_axis_gripper)
print(f"\nPolar coordinates (spherical, gripper frame):")
print(f"  θ (elevation from +Z): {theta_g:6.2f}°")
print(f"  φ (azimuth in XY):     {phi_g:6.2f}°")

# Plane angles (gripper frame)
xy_angle_g, xz_angle_g, yz_angle_g = calculate_plane_angles(mean_axis_gripper)
print(f"\nCoordinate plane angles (gripper frame):")
print(f"  XY plane (from +X): {xy_angle_g:6.2f}°")
print(f"  XZ plane (from +X): {xz_angle_g:6.2f}°")
print(f"  YZ plane (from +Y): {yz_angle_g:6.2f}°")

# Euler angles (gripper frame, camera forward is -Z)
roll_g, pitch_g, yaw_g = calculate_euler_angles(mean_axis_gripper, np.array([0, 0, -1]))
print(f"\nEuler angles (XYZ convention, rotation from camera -Z):")
print(f"  Roll  (X): {roll_g:6.2f}°")
print(f"  Pitch (Y): {pitch_g:6.2f}°")
print(f"  Yaw   (Z): {yaw_g:6.2f}°")

# Calculate MuJoCo axis-angle representation
local_forward = np.array([0, 0, -1])
rotation_axis = np.cross(local_forward, mean_axis_gripper)
rotation_axis_norm = np.linalg.norm(rotation_axis)

if rotation_axis_norm < 1e-6:
    if np.dot(local_forward, mean_axis_gripper) > 0:
        rotation = R.from_euler('xyz', [0, 0, 0])
    else:
        rotation = R.from_euler('xyz', [np.pi, 0, 0])
else:
    rotation_axis = rotation_axis / rotation_axis_norm
    angle = np.arccos(np.clip(np.dot(local_forward, mean_axis_gripper), -1, 1))
    rotation = R.from_rotvec(rotation_axis * angle)

rotvec = rotation.as_rotvec()
angle_rad = np.linalg.norm(rotvec)
if angle_rad > 1e-6:
    axis = rotvec / angle_rad
else:
    axis = np.array([1, 0, 0])
    angle_rad = 0

print(f"\n{'='*80}")
print("MUJOCO AXIS-ANGLE REPRESENTATION")
print(f"{'='*80}")

print(f"\nCamera orientation (axis-angle):")
print(f"  Axis:  [{axis[0]:8.6f}, {axis[1]:8.6f}, {axis[2]:8.6f}]")
print(f"  Angle: {angle_rad:.6f} rad = {np.degrees(angle_rad):.2f}°")

camera_pos = np.array([0.0025, 0.0609, 0.0120])

print(f"\n{'='*80}")
print("CAMERA CONFIGURATION")
print(f"{'='*80}")

print(f"\n<camera name=\"wrist_camera\"")
print(f"        pos=\"{camera_pos[0]:.4f} {camera_pos[1]:.4f} {camera_pos[2]:.4f}\"")
print(f"        axisangle=\"{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f} {angle_rad:.6f}\"")
print(f"        fovy=\"140\"/>")

print(f"\n{'='*80}")
