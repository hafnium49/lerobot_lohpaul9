#!/usr/bin/env python3
"""
Check if the 4 screw hole centers lie on a single plane (are coplanar).
"""

import numpy as np


# The 4 corner screw hole centers in STL frame (from previous analysis)
screw_holes_stl = np.array([
    [-0.0124, -0.0611, -0.0107],  # H1 (bottom-left)
    [-0.0131, -0.0858,  0.0019],  # H4 (top-left)
    [ 0.0172, -0.0616, -0.0110],  # H6 (bottom-right)
    [ 0.0181, -0.0859,  0.0019],  # H9 (top-right)
])

labels = ["H1 (bottom-left)", "H4 (top-left)", "H6 (bottom-right)", "H9 (top-right)"]

print("="*80)
print("Coplanarity Analysis of 4 Screw Hole Centers")
print("="*80)

print("\nScrew hole centers (STL frame):")
for i, (pos, label) in enumerate(zip(screw_holes_stl, labels)):
    print(f"  {label}: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}] m")

# Method 1: Check if 4 points are coplanar using determinant
# For 4 points P0, P1, P2, P3, they are coplanar if:
# det([P1-P0, P2-P0, P3-P0]) = 0

P0 = screw_holes_stl[0]
P1 = screw_holes_stl[1]
P2 = screw_holes_stl[2]
P3 = screw_holes_stl[3]

v1 = P1 - P0
v2 = P2 - P0
v3 = P3 - P0

# Create matrix and compute determinant
matrix = np.array([v1, v2, v3])
det = np.linalg.det(matrix)

print(f"\n{'='*80}")
print("Method 1: Determinant Test")
print(f"{'='*80}")
print(f"\nVectors from P0 to other points:")
print(f"  v1 (P0→P1): [{v1[0]:7.4f}, {v1[1]:7.4f}, {v1[2]:7.4f}]")
print(f"  v2 (P0→P2): [{v2[0]:7.4f}, {v2[1]:7.4f}, {v2[2]:7.4f}]")
print(f"  v3 (P0→P3): [{v3[0]:7.4f}, {v3[1]:7.4f}, {v3[2]:7.4f}]")
print(f"\nDeterminant = {det:.10f}")
print(f"Determinant magnitude = {abs(det):.10f}")

if abs(det) < 1e-6:
    print("\n✓ Points are COPLANAR (det ≈ 0)")
else:
    print(f"\n✗ Points are NOT perfectly coplanar (det = {det:.2e})")
    print(f"  Deviation from coplanarity: {abs(det):.2e} m³")

# Method 2: Fit a plane and measure distances
# Fit plane using SVD: find normal to the best-fit plane

# Center the points
centroid = screw_holes_stl.mean(axis=0)
centered_points = screw_holes_stl - centroid

# SVD to find the plane
U, S, Vt = np.linalg.svd(centered_points)

# The plane normal is the last row of Vt (smallest singular value)
plane_normal = Vt[-1, :]

# Ensure normal points in +Y direction (outward from fixed jaw)
if plane_normal[1] < 0:
    plane_normal = -plane_normal

# The plane equation: normal · (p - centroid) = 0
# Distance from point to plane: |normal · (p - centroid)|

distances = np.abs(np.dot(centered_points, plane_normal))

print(f"\n{'='*80}")
print("Method 2: Plane Fitting (SVD)")
print(f"{'='*80}")

print(f"\nCentroid: [{centroid[0]:7.4f}, {centroid[1]:7.4f}, {centroid[2]:7.4f}] m")
print(f"\nBest-fit plane normal: [{plane_normal[0]:7.4f}, {plane_normal[1]:7.4f}, {plane_normal[2]:7.4f}]")
print(f"Plane normal magnitude: {np.linalg.norm(plane_normal):.6f}")

print(f"\nSingular values:")
print(f"  S[0] = {S[0]:.6e} (largest variation)")
print(f"  S[1] = {S[1]:.6e} (medium variation)")
print(f"  S[2] = {S[2]:.6e} (smallest variation - perpendicular to plane)")

print(f"\nDistances from best-fit plane:")
for i, (dist, label) in enumerate(zip(distances, labels)):
    print(f"  {label}: {dist*1000:.4f} mm")

max_dist = distances.max()
print(f"\nMaximum distance from plane: {max_dist*1000:.4f} mm")

if max_dist < 1e-4:  # 0.1 mm
    print("\n✓ Points are essentially COPLANAR (max dist < 0.1 mm)")
elif max_dist < 1e-3:  # 1 mm
    print(f"\n≈ Points are nearly coplanar (max dist = {max_dist*1000:.2f} mm)")
else:
    print(f"\n✗ Points have significant non-coplanarity (max dist = {max_dist*1000:.2f} mm)")

# Method 3: Check angles between normal vectors
# If coplanar, all cross products should align with the same normal

print(f"\n{'='*80}")
print("Method 3: Cross Product Consistency")
print(f"{'='*80}")

# Calculate normals from different point triplets
n1 = np.cross(v1, v2)  # P0, P1, P2
n2 = np.cross(v2, v3)  # P0, P2, P3
n3 = np.cross(v1, v3)  # P0, P1, P3

# Normalize
n1_norm = n1 / np.linalg.norm(n1) if np.linalg.norm(n1) > 1e-10 else n1
n2_norm = n2 / np.linalg.norm(n2) if np.linalg.norm(n2) > 1e-10 else n2
n3_norm = n3 / np.linalg.norm(n3) if np.linalg.norm(n3) > 1e-10 else n3

print(f"\nNormals from different point triplets:")
print(f"  n1 (P0,P1,P2): [{n1_norm[0]:7.4f}, {n1_norm[1]:7.4f}, {n1_norm[2]:7.4f}]")
print(f"  n2 (P0,P2,P3): [{n2_norm[0]:7.4f}, {n2_norm[1]:7.4f}, {n2_norm[2]:7.4f}]")
print(f"  n3 (P0,P1,P3): [{n3_norm[0]:7.4f}, {n3_norm[1]:7.4f}, {n3_norm[2]:7.4f}]")

# Calculate angles between normals
angle_12 = np.degrees(np.arccos(np.clip(np.dot(n1_norm, n2_norm), -1, 1)))
angle_23 = np.degrees(np.arccos(np.clip(np.dot(n2_norm, n3_norm), -1, 1)))
angle_13 = np.degrees(np.arccos(np.clip(np.dot(n1_norm, n3_norm), -1, 1)))

print(f"\nAngles between normals:")
print(f"  n1 ↔ n2: {angle_12:.4f}°")
print(f"  n2 ↔ n3: {angle_23:.4f}°")
print(f"  n1 ↔ n3: {angle_13:.4f}°")

max_angle = max(angle_12, angle_23, angle_13)

if max_angle < 1.0:
    print(f"\n✓ Normals are consistent (max angle = {max_angle:.2f}°) - points are COPLANAR")
elif max_angle < 5.0:
    print(f"\n≈ Normals are nearly consistent (max angle = {max_angle:.2f}°) - nearly coplanar")
else:
    print(f"\n✗ Normals vary significantly (max angle = {max_angle:.2f}°) - NOT coplanar")

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

print(f"\nDeterminant: {abs(det):.2e} m³")
print(f"Max distance from best-fit plane: {max_dist*1000:.4f} mm")
print(f"Max angle between normals: {max_angle:.4f}°")

if abs(det) < 1e-6 and max_dist < 1e-4 and max_angle < 1.0:
    print("\n✓ CONCLUSION: The 4 screw holes are COPLANAR")
    print("\nThe mounting surface is a well-defined plane.")
    print(f"Plane normal (outward from jaw): [{plane_normal[0]:7.4f}, {plane_normal[1]:7.4f}, {plane_normal[2]:7.4f}]")
elif abs(det) < 1e-5 and max_dist < 1e-3 and max_angle < 5.0:
    print("\n≈ CONCLUSION: The 4 screw holes are NEARLY coplanar")
    print(f"\nSmall deviations likely due to manufacturing tolerances or STL mesh discretization.")
    print(f"Best-fit plane normal: [{plane_normal[0]:7.4f}, {plane_normal[1]:7.4f}, {plane_normal[2]:7.4f}]")
else:
    print("\n✗ CONCLUSION: The 4 screw holes have significant non-coplanarity")
    print("\nThis may indicate a curved mounting surface or data extraction issues.")

print(f"\n{'='*80}")
