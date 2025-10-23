#!/usr/bin/env python3
"""
Find the 4 screw holes at the corners of the mounting surface.
These are circular holes for mounting screws at each corner of the 32x32mm square.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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
print("Finding 4 Screw Holes at Mounting Surface Corners")
print("="*80)

# Read STL
stl_file = "src/lerobot/envs/so101_assets/official_model/assets/Wrist_Cam_Mount_32x32_UVC_Module_SO101_MIRRORED.stl"
print(f"\nReading: {stl_file}")
normals, vertices = read_stl_binary(stl_file)

# Convert mm to meters
vertices = vertices * 0.001

print(f"Total triangles: {len(vertices)}")

# Strategy: Look for cylindrical holes at the mounting surface
# Holes will have:
# 1. Normals pointing radially inward (perpendicular to Y-axis in cylindrical pattern)
# 2. Located at the tip of the mount (negative Y)
# 3. Vertices forming circles in XZ plane at similar Y positions

# Find triangles at the tip
all_vertices = vertices.reshape(-1, 3)
y_min = all_vertices[:, 1].min()
tip_threshold = y_min + 0.030  # Within 30mm of tip

print(f"\nStep 1: Find triangles at tip (Y < {tip_threshold:.4f} m)")

tip_triangles = []
tip_normals = []
tip_indices = []

for i, (tri_verts, normal) in enumerate(zip(vertices, normals)):
    tri_y_min = tri_verts[:, 1].min()
    if tri_y_min < tip_threshold:
        norm = normal / np.linalg.norm(normal)
        tip_triangles.append(tri_verts)
        tip_normals.append(norm)
        tip_indices.append(i)

print(f"  Found {len(tip_triangles)} triangles at tip")

tip_triangles = np.array(tip_triangles)
tip_normals = np.array(tip_normals)

# Look for hole geometry: triangles with normals that have small Y component
# (pointing radially, not outward from surface)
print(f"\nStep 2: Find hole triangles (normals perpendicular to Y-axis)")

hole_triangles = []
hole_normals = []

for tri_verts, norm in zip(tip_triangles, tip_normals):
    # Hole walls have normals mostly in XZ plane (small Y component)
    if abs(norm[1]) < 0.5:  # Y component < 0.5 (not pointing along Y)
        # Also check if near the front surface (Y > -0.080)
        tri_y_max = tri_verts[:, 1].max()
        if tri_y_max > -0.080:
            hole_triangles.append(tri_verts)
            hole_normals.append(norm)

print(f"  Found {len(hole_triangles)} potential hole triangles")

if len(hole_triangles) == 0:
    print("❌ No hole triangles found!")
    exit(1)

hole_triangles = np.array(hole_triangles)
hole_normals = np.array(hole_normals)

# Analyze hole triangle positions
hole_verts = hole_triangles.reshape(-1, 3)
print(f"\nHole vertices range:")
print(f"  X: [{hole_verts[:, 0].min():.4f}, {hole_verts[:, 0].max():.4f}] m")
print(f"  Y: [{hole_verts[:, 1].min():.4f}, {hole_verts[:, 1].max():.4f}] m")
print(f"  Z: [{hole_verts[:, 2].min():.4f}, {hole_verts[:, 2].max():.4f}] m")

# Cluster vertices in XZ plane to find hole centers
# Use a simple approach: look at unique vertex positions and cluster them
print(f"\nStep 3: Clustering vertices to find hole centers...")

# Get unique vertices (approximately)
unique_verts = []
for v in hole_verts:
    # Check if this vertex is far enough from existing unique vertices
    is_new = True
    for uv in unique_verts:
        if np.linalg.norm(v - uv) < 0.001:  # Within 1mm
            is_new = False
            break
    if is_new:
        unique_verts.append(v)

print(f"  Found ~{len(unique_verts)} unique vertices")

# Cluster by XZ position (ignore Y for now)
# We expect 4 clusters (one per hole) plus possibly the center dodecagon hole
from scipy.cluster.hierarchy import fclusterdata

# Extract XZ positions only
xz_positions = np.array([[v[0], v[2]] for v in unique_verts])

# Cluster with a threshold based on expected hole separation
# For a 32mm square with holes near corners, they'd be ~25-28mm apart
cluster_threshold = 0.008  # 8mm - holes within 8mm are same hole
clusters = fclusterdata(xz_positions, cluster_threshold, criterion='distance', method='complete')

n_clusters = clusters.max()
print(f"  Found {n_clusters} hole clusters")

# Calculate center of each cluster
hole_centers = []
for cluster_id in range(1, n_clusters + 1):
    cluster_mask = clusters == cluster_id
    cluster_verts = np.array([unique_verts[i] for i in range(len(unique_verts)) if cluster_mask[i]])

    center_x = cluster_verts[:, 0].mean()
    center_y = cluster_verts[:, 1].mean()
    center_z = cluster_verts[:, 2].mean()

    hole_centers.append([center_x, center_y, center_z])

    # Calculate hole size
    x_range = cluster_verts[:, 0].max() - cluster_verts[:, 0].min()
    z_range = cluster_verts[:, 2].max() - cluster_verts[:, 2].min()
    diameter = max(x_range, z_range)

    print(f"\n  Hole {cluster_id}:")
    print(f"    Center (STL): [{center_x:.4f}, {center_y:.4f}, {center_z:.4f}] m")
    print(f"    Size: {diameter*1000:.1f} mm diameter")
    print(f"    Vertices in cluster: {len(cluster_verts)}")

hole_centers = np.array(hole_centers)

# Identify which holes are screw holes (at corners) vs center lens hole
# Screw holes should be in 4 corners, forming a square pattern
# Center hole should be near the geometric center

if len(hole_centers) >= 4:
    print(f"\nStep 4: Identifying screw holes vs lens hole...")

    # Calculate geometric center of all holes
    center_xz = hole_centers[:, [0, 2]].mean(axis=0)
    print(f"  Geometric center (XZ): [{center_xz[0]:.4f}, {center_xz[1]:.4f}]")

    # Calculate distance of each hole from geometric center
    distances = []
    for i, hole in enumerate(hole_centers):
        dist = np.linalg.norm(hole[[0, 2]] - center_xz)
        distances.append(dist)

    distances = np.array(distances)

    # Screw holes should be at similar distances from center (corners of square)
    # Center lens hole should be closest to center

    # Sort by distance
    sorted_indices = np.argsort(distances)

    print(f"\n  Holes sorted by distance from center:")
    for rank, idx in enumerate(sorted_indices):
        hole = hole_centers[idx]
        dist = distances[idx]
        print(f"    {rank+1}. Hole {idx+1}: distance = {dist*1000:.1f} mm")

    # If we have 5 holes, first is center lens hole, remaining 4 are screw holes
    # If we have 4 holes, all are likely screw holes

    if len(hole_centers) == 5:
        lens_hole_idx = sorted_indices[0]
        screw_hole_indices = sorted_indices[1:]
        print(f"\n  ✅ Center lens hole: Hole {lens_hole_idx+1}")
        print(f"  ✅ Screw holes: Holes {[i+1 for i in screw_hole_indices]}")
    elif len(hole_centers) == 4:
        screw_hole_indices = sorted_indices
        lens_hole_idx = None
        print(f"\n  ⚠️  Found 4 holes - assuming all are screw holes")
        print(f"  ⚠️  Center lens hole not detected (may not be modeled as hole)")
        # Estimate lens hole position as geometric center
        estimated_lens_center = [center_xz[0], hole_centers[:, 1].mean(), center_xz[1]]
        print(f"  📍 Estimated lens hole center: [{estimated_lens_center[0]:.4f}, {estimated_lens_center[1]:.4f}, {estimated_lens_center[2]:.4f}]")
    else:
        screw_hole_indices = list(range(len(hole_centers)))
        lens_hole_idx = None
        print(f"\n  ⚠️  Found {len(hole_centers)} holes")

    # Calculate mounting surface bounds from screw holes
    screw_holes = hole_centers[screw_hole_indices]

    print(f"\n" + "="*80)
    print("SCREW HOLE POSITIONS")
    print("="*80)

    for i, hole_idx in enumerate(screw_hole_indices):
        hole = hole_centers[hole_idx]
        print(f"\nScrew Hole {i+1} (Cluster {hole_idx+1}):")
        print(f"  STL frame:     [{hole[0]:7.4f}, {hole[1]:7.4f}, {hole[2]:7.4f}] m")

        # Transform to gripper frame
        mount_pos = np.array([0.0, -0.000218214, 0.000949706])
        hole_rotated = np.array([hole[0], -hole[1], -hole[2]])
        hole_gripper = mount_pos + hole_rotated
        print(f"  Gripper frame: [{hole_gripper[0]:7.4f}, {hole_gripper[1]:7.4f}, {hole_gripper[2]:7.4f}] m")

    # Calculate mounting surface center from screw holes
    screw_center_x = screw_holes[:, 0].mean()
    screw_center_y = screw_holes[:, 1].mean()
    screw_center_z = screw_holes[:, 2].mean()

    print(f"\n" + "="*80)
    print("MOUNTING SURFACE CENTER (from screw hole average)")
    print("="*80)
    print(f"\nSTL frame:     [{screw_center_x:.4f}, {screw_center_y:.4f}, {screw_center_z:.4f}] m")

    mount_pos = np.array([0.0, -0.000218214, 0.000949706])
    center_rotated = np.array([screw_center_x, -screw_center_y, -screw_center_z])
    center_gripper = mount_pos + center_rotated
    print(f"Gripper frame: [{center_gripper[0]:.4f}, {center_gripper[1]:.4f}, {center_gripper[2]:.4f}] m")

    # Calculate mounting surface dimensions
    x_span = screw_holes[:, 0].max() - screw_holes[:, 0].min()
    z_span = screw_holes[:, 2].max() - screw_holes[:, 2].min()

    print(f"\nScrew hole span:")
    print(f"  X: {x_span*1000:.1f} mm")
    print(f"  Z: {z_span*1000:.1f} mm")

    # For 32mm mounting surface with M2.5 or M3 screws, holes are typically
    # on 28-29mm centers (3-4mm from edges)
    estimated_surface_size = max(x_span, z_span) + 0.006  # Add ~6mm (2*3mm margin)
    print(f"  Estimated surface size: {estimated_surface_size*1000:.1f} mm square")

# Create visualization
print("\nGenerating visualization...")
fig = plt.figure(figsize=(18, 12))

# View 1: 3D view of holes
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.set_title('3D View: Hole Triangles', fontsize=10)

hole_mesh = Poly3DCollection(hole_triangles, alpha=0.6, edgecolor='k', linewidths=0.3)
hole_mesh.set_facecolor([1, 0.5, 0])
ax1.add_collection3d(hole_mesh)

# Plot hole centers
for i, center in enumerate(hole_centers):
    ax1.scatter([center[0]], [center[1]], [center[2]], c='red', s=100, marker='o')
    ax1.text(center[0], center[1], center[2], f'  H{i+1}', fontsize=8)

ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')

# View 2: Top-down view (XZ plane)
ax2 = fig.add_subplot(2, 3, 2)
ax2.set_title('Top-Down View: Hole Centers (XZ plane)', fontsize=10)

for i, center in enumerate(hole_centers):
    ax2.scatter(center[0]*1000, center[2]*1000, c='red', s=200, marker='o', zorder=10)
    ax2.text(center[0]*1000 + 1, center[2]*1000 + 1, f'H{i+1}', fontsize=10)

# Draw screw hole pattern
if len(hole_centers) >= 4:
    screw_xz = screw_holes[:, [0, 2]] * 1000
    # Draw lines connecting screw holes
    for i in range(len(screw_xz)):
        for j in range(i+1, len(screw_xz)):
            ax2.plot([screw_xz[i,0], screw_xz[j,0]],
                    [screw_xz[i,1], screw_xz[j,1]],
                    'b--', alpha=0.3, linewidth=1)

    # Draw bounding box
    x_min, x_max = screw_holes[:, 0].min(), screw_holes[:, 0].max()
    z_min, z_max = screw_holes[:, 2].min(), screw_holes[:, 2].max()
    ax2.plot([x_min*1000, x_max*1000, x_max*1000, x_min*1000, x_min*1000],
            [z_min*1000, z_min*1000, z_max*1000, z_max*1000, z_min*1000],
            'g-', linewidth=2, label='Screw hole bounds')

    # Mark center
    ax2.scatter([screw_center_x*1000], [screw_center_z*1000],
               c='blue', s=300, marker='*', zorder=15, label='Center')

ax2.set_xlabel('X (mm)')
ax2.set_ylabel('Z (mm)')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.legend()

# View 3: Side view (XY plane)
ax3 = fig.add_subplot(2, 3, 3)
ax3.set_title('Side View: Hole Centers (XY plane)', fontsize=10)

for i, center in enumerate(hole_centers):
    ax3.scatter(center[0]*1000, center[1]*1000, c='red', s=200, marker='o')
    ax3.text(center[0]*1000 + 1, center[1]*1000 - 1, f'H{i+1}', fontsize=10)

ax3.set_xlabel('X (mm)')
ax3.set_ylabel('Y (mm)')
ax3.grid(True, alpha=0.3)

# View 4: Front view (YZ plane)
ax4 = fig.add_subplot(2, 3, 4)
ax4.set_title('Front View: Hole Centers (YZ plane)', fontsize=10)

for i, center in enumerate(hole_centers):
    ax4.scatter(center[1]*1000, center[2]*1000, c='red', s=200, marker='o')
    ax4.text(center[1]*1000 + 1, center[2]*1000 + 1, f'H{i+1}', fontsize=10)

ax4.set_xlabel('Y (mm)')
ax4.set_ylabel('Z (mm)')
ax4.grid(True, alpha=0.3)

# View 5: Hole triangles top-down
ax5 = fig.add_subplot(2, 3, 5)
ax5.set_title('Top-Down: All Hole Triangles', fontsize=10)

for tri in hole_triangles:
    triangle = np.vstack([tri, tri[0]])
    ax5.plot(triangle[:, 0]*1000, triangle[:, 2]*1000, 'orange', linewidth=0.5, alpha=0.3)

for i, center in enumerate(hole_centers):
    ax5.scatter(center[0]*1000, center[2]*1000, c='red', s=200, marker='o', zorder=10)
    ax5.text(center[0]*1000 + 1, center[2]*1000 + 1, f'H{i+1}', fontsize=10)

ax5.set_xlabel('X (mm)')
ax5.set_ylabel('Z (mm)')
ax5.set_aspect('equal')
ax5.grid(True, alpha=0.3)

# View 6: Summary text
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')

if len(hole_centers) >= 4:
    info_text = f"""
SCREW HOLE ANALYSIS

Found {len(hole_centers)} holes total
Screw holes: {len(screw_hole_indices)}

Screw Hole Centers (STL frame):
"""
    for i, hole_idx in enumerate(screw_hole_indices):
        hole = hole_centers[hole_idx]
        info_text += f"  H{hole_idx+1}: [{hole[0]:6.3f}, {hole[1]:6.3f}, {hole[2]:6.3f}]\n"

    info_text += f"""
Mounting Surface Center:
  STL: [{screw_center_x:.4f}, {screw_center_y:.4f}, {screw_center_z:.4f}]
  Gripper: [{center_gripper[0]:.4f}, {center_gripper[1]:.4f}, {center_gripper[2]:.4f}]

Screw Hole Span:
  X: {x_span*1000:.1f} mm
  Z: {z_span*1000:.1f} mm

Estimated Surface: {estimated_surface_size*1000:.1f} mm square

Camera Position (Gripper):
  pos="{center_gripper[0]:.4f} {center_gripper[1]:.4f} {center_gripper[2]:.4f}"
"""
else:
    info_text = f"Found {len(hole_centers)} holes\n(Need at least 4 for analysis)"

ax6.text(0.05, 0.5, info_text, fontsize=9, family='monospace',
         verticalalignment='center', transform=ax6.transAxes)

plt.tight_layout()

output_file = "screw_holes_analysis.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✅ Saved: {output_file}")
print("="*80)
