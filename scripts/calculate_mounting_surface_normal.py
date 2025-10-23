#!/usr/bin/env python3
"""
Calculate the mounting surface plane normal from 4 screw hole centers.
This normal represents the direction the camera should face (outward from fixed jaw).
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


# The 4 corner screw hole centers in STL frame
screw_holes_stl = np.array([
    [-0.0124, -0.0611, -0.0107],  # H1 (bottom-left)
    [-0.0131, -0.0858,  0.0019],  # H4 (top-left)
    [ 0.0172, -0.0616, -0.0110],  # H6 (bottom-right)
    [ 0.0181, -0.0859,  0.0019],  # H9 (top-right)
])

labels = ["H1 (bottom-left)", "H4 (top-left)", "H6 (bottom-right)", "H9 (top-right)"]

print("="*80)
print("Mounting Surface Plane Normal Calculation")
print("="*80)

print("\nScrew hole centers (STL frame):")
for i, (pos, label) in enumerate(zip(screw_holes_stl, labels)):
    print(f"  {label}: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}] m")

# Step 1: Calculate centroid
centroid = screw_holes_stl.mean(axis=0)
print(f"\nCentroid: [{centroid[0]:7.4f}, {centroid[1]:7.4f}, {centroid[2]:7.4f}] m")

# Step 2: Center the points
centered_points = screw_holes_stl - centroid

# Step 3: Use SVD to find the best-fit plane
# The plane normal is the singular vector with smallest singular value
U, S, Vt = np.linalg.svd(centered_points)

# The plane normal is the last row of Vt
plane_normal_stl = Vt[-1, :]

print(f"\n{'='*80}")
print("Plane Fitting (SVD)")
print(f"{'='*80}")

print(f"\nSingular values:")
print(f"  S[0] = {S[0]:.6e} (largest - variation along plane)")
print(f"  S[1] = {S[1]:.6e} (medium - variation along plane)")
print(f"  S[2] = {S[2]:.6e} (smallest - variation perpendicular to plane)")

print(f"\nPlane normal (raw): [{plane_normal_stl[0]:7.4f}, {plane_normal_stl[1]:7.4f}, {plane_normal_stl[2]:7.4f}]")

# Step 4: Ensure normal points outward (positive Y component in STL frame)
# In STL frame, mounting surface faces toward +Y (outward from fixed jaw)
if plane_normal_stl[1] < 0:
    plane_normal_stl = -plane_normal_stl
    print(f"  → Flipped to point in +Y direction")

print(f"\nPlane normal (STL frame): [{plane_normal_stl[0]:7.4f}, {plane_normal_stl[1]:7.4f}, {plane_normal_stl[2]:7.4f}]")

# Verify distances from plane
distances = np.abs(np.dot(centered_points, plane_normal_stl))
print(f"\nDistances from best-fit plane:")
for i, (dist, label) in enumerate(zip(distances, labels)):
    print(f"  {label}: {dist*1000:.4f} mm")
print(f"Maximum distance: {distances.max()*1000:.4f} mm")

# Step 5: Transform to gripper frame (180° rotation around Y-axis)
# Transformation: [x, y, z]_STL → [x, -y, -z]_gripper
plane_normal_gripper = np.array([
    plane_normal_stl[0],
    -plane_normal_stl[1],
    -plane_normal_stl[2]
])

print(f"\n{'='*80}")
print("Transformation to Gripper Frame")
print(f"{'='*80}")

print(f"\nPlane normal (gripper frame): [{plane_normal_gripper[0]:7.4f}, {plane_normal_gripper[1]:7.4f}, {plane_normal_gripper[2]:7.4f}]")

# Analyze orientation in gripper frame
angle_from_x = np.degrees(np.arccos(np.clip(abs(plane_normal_gripper[0]), 0, 1)))
angle_from_y = np.degrees(np.arccos(np.clip(abs(plane_normal_gripper[1]), 0, 1)))
angle_from_z = np.degrees(np.arccos(np.clip(abs(plane_normal_gripper[2]), 0, 1)))

print(f"\nAngles from coordinate axes (gripper frame):")
print(f"  From ±X: {angle_from_x:.2f}°")
print(f"  From ±Y: {angle_from_y:.2f}°")
print(f"  From ±Z: {angle_from_z:.2f}°")

# Step 6: Convert to MuJoCo axis-angle representation
# Camera's local forward direction is -Z axis
# We need to rotate from [0, 0, -1] to plane_normal_gripper

print(f"\n{'='*80}")
print("MuJoCo Axis-Angle Representation")
print(f"{'='*80}")

local_forward = np.array([0, 0, -1])  # Camera looks down -Z in its local frame
target_direction = plane_normal_gripper

print(f"\nCamera local forward: [{local_forward[0]:7.4f}, {local_forward[1]:7.4f}, {local_forward[2]:7.4f}]")
print(f"Target direction:     [{target_direction[0]:7.4f}, {target_direction[1]:7.4f}, {target_direction[2]:7.4f}]")

# Calculate rotation axis and angle
rotation_axis = np.cross(local_forward, target_direction)
rotation_axis_norm = np.linalg.norm(rotation_axis)

if rotation_axis_norm < 1e-6:
    # Parallel or anti-parallel
    dot_product = np.dot(local_forward, target_direction)
    if dot_product > 0:
        # Already aligned
        print("\n✓ Camera already aligned with target direction")
        axis = np.array([1, 0, 0])
        angle_rad = 0.0
    else:
        # 180° rotation needed
        print("\n⚠ 180° rotation needed")
        axis = np.array([1, 0, 0])  # Arbitrary axis for 180° rotation
        angle_rad = np.pi
else:
    # General case
    rotation_axis = rotation_axis / rotation_axis_norm
    angle_rad = np.arccos(np.clip(np.dot(local_forward, target_direction), -1, 1))

    # Use scipy for robust rotation
    rotation = R.from_rotvec(rotation_axis * angle_rad)
    rotvec = rotation.as_rotvec()
    angle_rad = np.linalg.norm(rotvec)

    if angle_rad > 1e-6:
        axis = rotvec / angle_rad
    else:
        axis = np.array([1, 0, 0])
        angle_rad = 0.0

angle_deg = np.degrees(angle_rad)

print(f"\nRotation:")
print(f"  Axis:  [{axis[0]:8.6f}, {axis[1]:8.6f}, {axis[2]:8.6f}]")
print(f"  Angle: {angle_rad:.6f} rad = {angle_deg:.2f}°")

# Verify the rotation
rotation_check = R.from_rotvec(axis * angle_rad)
rotated_forward = rotation_check.apply(local_forward)
alignment_error = np.linalg.norm(rotated_forward - target_direction)

print(f"\nVerification:")
print(f"  Rotated forward: [{rotated_forward[0]:7.4f}, {rotated_forward[1]:7.4f}, {rotated_forward[2]:7.4f}]")
print(f"  Target:          [{target_direction[0]:7.4f}, {target_direction[1]:7.4f}, {target_direction[2]:7.4f}]")
print(f"  Alignment error: {alignment_error:.6e}")

if alignment_error < 1e-3:
    print("  ✓ Rotation verified!")
else:
    print(f"  ⚠ Warning: alignment error = {alignment_error:.2e}")

# Step 7: Output camera configuration
camera_pos = np.array([0.0025, 0.0609, 0.0120])  # From dodecagon hole center

print(f"\n{'='*80}")
print("CAMERA CONFIGURATION")
print(f"{'='*80}")

print(f"\nPosition (from dodecagon hole analysis):")
print(f"  pos=\"{camera_pos[0]:.4f} {camera_pos[1]:.4f} {camera_pos[2]:.4f}\"")

print(f"\nOrientation (from mounting surface normal):")
print(f"  axisangle=\"{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f} {angle_rad:.6f}\"")

print(f"\n{'='*80}")
print("XML Configuration")
print(f"{'='*80}")

print(f"\n<camera name=\"wrist_camera\"")
print(f"        pos=\"{camera_pos[0]:.4f} {camera_pos[1]:.4f} {camera_pos[2]:.4f}\"")
print(f"        axisangle=\"{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f} {angle_rad:.6f}\"")
print(f"        fovy=\"140\"/>")

# Additional information
print(f"\n{'='*80}")
print("Summary")
print(f"{'='*80}")

print(f"\nMounting surface normal (STL):     [{plane_normal_stl[0]:7.4f}, {plane_normal_stl[1]:7.4f}, {plane_normal_stl[2]:7.4f}]")
print(f"Mounting surface normal (gripper): [{plane_normal_gripper[0]:7.4f}, {plane_normal_gripper[1]:7.4f}, {plane_normal_gripper[2]:7.4f}]")
print(f"Camera faces: Outward from fixed jaw mounting surface")
print(f"Rotation required: {angle_deg:.2f}° around axis [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")

print(f"\n{'='*80}")
