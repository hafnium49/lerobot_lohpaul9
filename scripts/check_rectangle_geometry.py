#!/usr/bin/env python3
"""
Check if the 4 screw hole centers form a rectangle.
Analyzes distances, angles, and geometric properties.
"""

import numpy as np


# The 4 corner screw hole centers in STL frame
screw_holes_stl = np.array([
    [-0.0124, -0.0611, -0.0107],  # H1 (bottom-left)
    [-0.0131, -0.0858,  0.0019],  # H4 (top-left)
    [ 0.0172, -0.0616, -0.0110],  # H6 (bottom-right)
    [ 0.0181, -0.0859,  0.0019],  # H9 (top-right)
])

labels = ["H1 (bottom-left)", "H4 (top-left)", "H6 (bottom-right)", "H9 (top-right)"]

print("="*80)
print("Rectangle Geometry Analysis of 4 Screw Hole Centers")
print("="*80)

print("\nScrew hole centers (STL frame):")
for i, (pos, label) in enumerate(zip(screw_holes_stl, labels)):
    print(f"  {label}: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}] m")

# For a rectangle, we expect:
# 1. 4 equal angles (all 90°)
# 2. Opposite sides equal in length
# 3. Diagonals equal in length

# Calculate all pairwise distances
print(f"\n{'='*80}")
print("Pairwise Distances")
print(f"{'='*80}")

distances = {}
for i in range(4):
    for j in range(i+1, 4):
        dist = np.linalg.norm(screw_holes_stl[i] - screw_holes_stl[j])
        key = f"{labels[i][:2]} ↔ {labels[j][:2]}"
        distances[key] = dist
        print(f"  {key}: {dist*1000:.4f} mm")

# Expected edges for rectangle (assuming layout):
# H1 (bottom-left) - H4 (top-left): left edge
# H1 (bottom-left) - H6 (bottom-right): bottom edge
# H4 (top-left) - H9 (top-right): top edge
# H6 (bottom-right) - H9 (top-right): right edge

# Diagonals:
# H1 - H9 and H4 - H6

edge_H1_H4 = np.linalg.norm(screw_holes_stl[0] - screw_holes_stl[1])  # left
edge_H1_H6 = np.linalg.norm(screw_holes_stl[0] - screw_holes_stl[2])  # bottom
edge_H4_H9 = np.linalg.norm(screw_holes_stl[1] - screw_holes_stl[3])  # top
edge_H6_H9 = np.linalg.norm(screw_holes_stl[2] - screw_holes_stl[3])  # right

diag_H1_H9 = np.linalg.norm(screw_holes_stl[0] - screw_holes_stl[3])
diag_H4_H6 = np.linalg.norm(screw_holes_stl[1] - screw_holes_stl[2])

print(f"\n{'='*80}")
print("Rectangle Edge Analysis")
print(f"{'='*80}")

print(f"\nEdges:")
print(f"  Left edge   (H1-H4): {edge_H1_H4*1000:.4f} mm")
print(f"  Right edge  (H6-H9): {edge_H6_H9*1000:.4f} mm")
print(f"  Bottom edge (H1-H6): {edge_H1_H6*1000:.4f} mm")
print(f"  Top edge    (H4-H9): {edge_H4_H9*1000:.4f} mm")

print(f"\nDiagonals:")
print(f"  Diagonal 1 (H1-H9): {diag_H1_H9*1000:.4f} mm")
print(f"  Diagonal 2 (H4-H6): {diag_H4_H6*1000:.4f} mm")

# Check if opposite sides are equal
left_right_diff = abs(edge_H1_H4 - edge_H6_H9)
top_bottom_diff = abs(edge_H4_H9 - edge_H1_H6)
diagonal_diff = abs(diag_H1_H9 - diag_H4_H6)

print(f"\n{'='*80}")
print("Opposite Side Comparison")
print(f"{'='*80}")

print(f"\nLeft vs Right edge difference: {left_right_diff*1000:.4f} mm")
print(f"Top vs Bottom edge difference: {top_bottom_diff*1000:.4f} mm")
print(f"Diagonal difference: {diagonal_diff*1000:.4f} mm")

tolerance_mm = 0.5  # 0.5mm tolerance

if left_right_diff*1000 < tolerance_mm:
    print(f"  ✓ Left and right edges are equal (within {tolerance_mm} mm)")
else:
    print(f"  ✗ Left and right edges differ by {left_right_diff*1000:.4f} mm")

if top_bottom_diff*1000 < tolerance_mm:
    print(f"  ✓ Top and bottom edges are equal (within {tolerance_mm} mm)")
else:
    print(f"  ✗ Top and bottom edges differ by {top_bottom_diff*1000:.4f} mm")

if diagonal_diff*1000 < tolerance_mm:
    print(f"  ✓ Diagonals are equal (within {tolerance_mm} mm)")
else:
    print(f"  ✗ Diagonals differ by {diagonal_diff*1000:.4f} mm")

# Check angles at corners
print(f"\n{'='*80}")
print("Corner Angles")
print(f"{'='*80}")

# Angle at H1: between vectors H1→H4 and H1→H6
v_H1_H4 = screw_holes_stl[1] - screw_holes_stl[0]
v_H1_H6 = screw_holes_stl[2] - screw_holes_stl[0]
angle_H1 = np.degrees(np.arccos(np.clip(np.dot(v_H1_H4, v_H1_H6) / (np.linalg.norm(v_H1_H4) * np.linalg.norm(v_H1_H6)), -1, 1)))

# Angle at H4: between vectors H4→H1 and H4→H9
v_H4_H1 = screw_holes_stl[0] - screw_holes_stl[1]
v_H4_H9 = screw_holes_stl[3] - screw_holes_stl[1]
angle_H4 = np.degrees(np.arccos(np.clip(np.dot(v_H4_H1, v_H4_H9) / (np.linalg.norm(v_H4_H1) * np.linalg.norm(v_H4_H9)), -1, 1)))

# Angle at H6: between vectors H6→H1 and H6→H9
v_H6_H1 = screw_holes_stl[0] - screw_holes_stl[2]
v_H6_H9 = screw_holes_stl[3] - screw_holes_stl[2]
angle_H6 = np.degrees(np.arccos(np.clip(np.dot(v_H6_H1, v_H6_H9) / (np.linalg.norm(v_H6_H1) * np.linalg.norm(v_H6_H9)), -1, 1)))

# Angle at H9: between vectors H9→H4 and H9→H6
v_H9_H4 = screw_holes_stl[1] - screw_holes_stl[3]
v_H9_H6 = screw_holes_stl[2] - screw_holes_stl[3]
angle_H9 = np.degrees(np.arccos(np.clip(np.dot(v_H9_H4, v_H9_H6) / (np.linalg.norm(v_H9_H4) * np.linalg.norm(v_H9_H6)), -1, 1)))

print(f"\nCorner angles:")
print(f"  Angle at H1 (bottom-left): {angle_H1:.2f}°")
print(f"  Angle at H4 (top-left):    {angle_H4:.2f}°")
print(f"  Angle at H6 (bottom-right): {angle_H6:.2f}°")
print(f"  Angle at H9 (top-right):   {angle_H9:.2f}°")

angles = [angle_H1, angle_H4, angle_H6, angle_H9]
mean_angle = np.mean(angles)
max_angle_dev = max([abs(a - 90) for a in angles])

print(f"\nAngle statistics:")
print(f"  Mean angle: {mean_angle:.2f}°")
print(f"  Max deviation from 90°: {max_angle_dev:.2f}°")

angle_tolerance = 2.0  # 2 degree tolerance

all_right_angles = all([abs(a - 90) < angle_tolerance for a in angles])

if all_right_angles:
    print(f"  ✓ All corners are right angles (within {angle_tolerance}°)")
else:
    print(f"  ✗ Some corners deviate from 90° by more than {angle_tolerance}°")

# Calculate aspect ratio
width = (edge_H1_H6 + edge_H4_H9) / 2
height = (edge_H1_H4 + edge_H6_H9) / 2
aspect_ratio = width / height

print(f"\n{'='*80}")
print("Rectangle Dimensions")
print(f"{'='*80}")

print(f"\nAverage width  (H1-H6, H4-H9): {width*1000:.4f} mm")
print(f"Average height (H1-H4, H6-H9): {height*1000:.4f} mm")
print(f"Aspect ratio (width/height): {aspect_ratio:.4f}")

# Check Pythagorean theorem: diagonal² = width² + height²
expected_diagonal = np.sqrt(width**2 + height**2)
actual_diagonal = (diag_H1_H9 + diag_H4_H6) / 2
pythagorean_error = abs(expected_diagonal - actual_diagonal)

print(f"\nPythagorean check:")
print(f"  Expected diagonal: {expected_diagonal*1000:.4f} mm")
print(f"  Actual diagonal:   {actual_diagonal*1000:.4f} mm")
print(f"  Error: {pythagorean_error*1000:.4f} mm")

if pythagorean_error*1000 < tolerance_mm:
    print(f"  ✓ Pythagorean theorem satisfied (within {tolerance_mm} mm)")
else:
    print(f"  ✗ Pythagorean theorem error: {pythagorean_error*1000:.4f} mm")

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

is_rectangle = (
    left_right_diff*1000 < tolerance_mm and
    top_bottom_diff*1000 < tolerance_mm and
    diagonal_diff*1000 < tolerance_mm and
    all_right_angles and
    pythagorean_error*1000 < tolerance_mm
)

print(f"\nOpposite sides equal: {'✓' if left_right_diff*1000 < tolerance_mm and top_bottom_diff*1000 < tolerance_mm else '✗'}")
print(f"Diagonals equal: {'✓' if diagonal_diff*1000 < tolerance_mm else '✗'}")
print(f"Right angles at corners: {'✓' if all_right_angles else '✗'}")
print(f"Pythagorean theorem: {'✓' if pythagorean_error*1000 < tolerance_mm else '✗'}")

if is_rectangle:
    print(f"\n✓ CONCLUSION: The 4 screw holes form a RECTANGLE")
    print(f"\n  Dimensions: {width*1000:.2f} mm × {height*1000:.2f} mm")
    print(f"  Aspect ratio: {aspect_ratio:.3f}")
else:
    print(f"\n✗ CONCLUSION: The 4 screw holes do NOT form a perfect rectangle")
    print(f"\n  But they may form a near-rectangle with manufacturing tolerances")

print(f"\n{'='*80}")
