#!/usr/bin/env python3
"""
Find and visualize the square mounting surface at the tip of camera mount.
The surface should have 5 holes: 4 screw holes at corners + 1 dodecagon lens hole in center.

New strategy: Look for all triangles with similar normals (forming a flat surface)
at the tip of the camera mount, regardless of exact Y position.
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
print("Finding and Visualizing Square Mounting Surface (v2)")
print("="*80)

# Read STL
stl_file = "src/lerobot/envs/so101_assets/official_model/assets/Wrist_Cam_Mount_32x32_UVC_Module_SO101_MIRRORED.stl"
print(f"\nReading: {stl_file}")
normals, vertices = read_stl_binary(stl_file)

# Convert mm to meters
vertices = vertices * 0.001

print(f"Total triangles: {len(vertices)}")

# Find tip of camera mount (most negative Y)
all_vertices = vertices.reshape(-1, 3)
y_min = all_vertices[:, 1].min()
y_max = all_vertices[:, 1].max()

print(f"\nY range: [{y_min:.4f}, {y_max:.4f}] m")
print(f"Tip at: Y = {y_min:.4f} m")

# Strategy: Find the dominant flat surface at the tip
# 1. Look for triangles at the tip region
# 2. Cluster by similar normal directions
# 3. Pick the largest cluster (should be the mounting surface)

tip_threshold = y_min + 0.030  # Within 30mm of tip
print(f"\nSearching for surfaces at Y < {tip_threshold:.4f} m...")

# Collect all triangles at tip with any +Y normal component
tip_triangles = []
tip_normals = []
tip_indices = []

for i, (tri_verts, normal) in enumerate(zip(vertices, normals)):
    tri_y_min = tri_verts[:, 1].min()
    if tri_y_min < tip_threshold:
        norm = normal / np.linalg.norm(normal)
        if norm[1] > 0.3:  # Has some +Y component
            tip_triangles.append(tri_verts)
            tip_normals.append(norm)
            tip_indices.append(i)

print(f"Found {len(tip_triangles)} triangles at tip with +Y normals")

if len(tip_triangles) == 0:
    print("❌ No triangles found!")
    exit(1)

tip_triangles = np.array(tip_triangles)
tip_normals = np.array(tip_normals)

# Cluster by normal direction
# Find the surface with normal pointing primarily in +Y direction (outward from mount)
print("\nLooking for surface with +Y pointing normal...")

# We expect the mounting surface normal to be around [0, 0.9, -0.4] (tilted but primarily +Y)
# The Y component should be larger than the Z component
# Filter for normals where Y > |Z|

valid_tip_triangles = []
valid_tip_normals = []

for tri_verts, norm in zip(tip_triangles, tip_normals):
    # Check if normal points primarily in +Y (not +Z or -Z)
    if norm[1] > 0.6 and norm[1] > abs(norm[2]) * 1.5:
        valid_tip_triangles.append(tri_verts)
        valid_tip_normals.append(norm)

print(f"Found {len(valid_tip_normals)} triangles with primarily +Y normals")

if len(valid_tip_normals) == 0:
    print("❌ No valid triangles found!")
    exit(1)

valid_tip_triangles = np.array(valid_tip_triangles)
valid_tip_normals = np.array(valid_tip_normals)

# Now find the dominant cluster among these
dominant_normal = None
max_cluster_size = 0

for i, normal_i in enumerate(valid_tip_normals):
    # Count how many normals are similar to this one
    angles = np.arccos(np.clip(np.dot(valid_tip_normals, normal_i), -1, 1))
    similar_count = np.sum(angles < np.radians(15))  # Within 15 degrees

    if similar_count > max_cluster_size:
        max_cluster_size = similar_count
        dominant_normal = normal_i

print(f"Dominant normal: [{dominant_normal[0]:.4f}, {dominant_normal[1]:.4f}, {dominant_normal[2]:.4f}]")
print(f"Cluster size: {max_cluster_size} triangles")

# Extract triangles with normals similar to dominant normal
mounting_surface_triangles = []
mounting_surface_normals = []

for tri_verts, norm in zip(valid_tip_triangles, valid_tip_normals):
    angle = np.arccos(np.clip(np.dot(norm, dominant_normal), -1, 1))
    if angle < np.radians(15):  # Within 15 degrees of dominant normal
        mounting_surface_triangles.append(tri_verts)
        mounting_surface_normals.append(norm)

print(f"\nFound {len(mounting_surface_triangles)} triangles in mounting surface")

if len(mounting_surface_triangles) == 0:
    print("❌ No mounting surface found!")
    exit(1)

mounting_surface_triangles = np.array(mounting_surface_triangles)
mounting_surface_normals = np.array(mounting_surface_normals)

# Analyze surface
surface_verts = mounting_surface_triangles.reshape(-1, 3)

x_min, x_max = surface_verts[:, 0].min(), surface_verts[:, 0].max()
y_min_surf, y_max_surf = surface_verts[:, 1].min(), surface_verts[:, 1].max()
z_min, z_max = surface_verts[:, 2].min(), surface_verts[:, 2].max()

width = (x_max - x_min) * 1000  # mm
height = (z_max - z_min) * 1000  # mm

print(f"\nMounting Surface:")
print(f"  X: [{x_min:.4f}, {x_max:.4f}] m (width: {width:.1f} mm)")
print(f"  Y: [{y_min_surf:.4f}, {y_max_surf:.4f}] m")
print(f"  Z: [{z_min:.4f}, {z_max:.4f}] m (height: {height:.1f} mm)")

# Check if square (~32x32mm)
if 30 < width < 35 and 30 < height < 35 and abs(width - height) < 3:
    print(f"  ✅ Square surface ~32x32mm confirmed!")
else:
    print(f"  ⚠️  Size: {width:.1f}mm x {height:.1f}mm (expected ~32x32mm)")

# Calculate center (dodecagon lens hole center)
center_x = (x_min + x_max) / 2
center_y = y_max_surf  # At the surface
center_z = (z_min + z_max) / 2

print(f"\nCenter of mounting surface (lens hole):")
print(f"  STL frame: [{center_x:.4f}, {center_y:.4f}, {center_z:.4f}] m")

# Average normal
avg_normal = np.mean(mounting_surface_normals, axis=0)
avg_normal = avg_normal / np.linalg.norm(avg_normal)

print(f"  Surface normal: [{avg_normal[0]:.4f}, {avg_normal[1]:.4f}, {avg_normal[2]:.4f}]")

tilt_angle = np.degrees(np.arccos(abs(avg_normal[1])))
if tilt_angle < 5:
    print(f"  ✅ Surface is flat (tilt: {tilt_angle:.1f}°)")
else:
    print(f"  ⚠️  Surface has {tilt_angle:.1f}° tilt from Y axis")

# Create visualization
print("\nGenerating visualization...")
fig = plt.figure(figsize=(16, 12))

# View 1: Full camera mount with highlighted mounting surface
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.set_title('Full Camera Mount\n(Red = Mounting Surface)', fontsize=10)

# Plot full mount (subsampled)
full_mesh = Poly3DCollection(vertices[::10], alpha=0.3, edgecolor='k', linewidths=0.1)
full_mesh.set_facecolor([0.7, 0.7, 0.7])
ax1.add_collection3d(full_mesh)

# Plot mounting surface (highlighted)
mount_mesh = Poly3DCollection(mounting_surface_triangles, alpha=0.9, edgecolor='darkred', linewidths=0.5)
mount_mesh.set_facecolor([1, 0, 0])
ax1.add_collection3d(mount_mesh)

# Plot center point
ax1.scatter([center_x], [center_y], [center_z], c='blue', s=100, marker='o', label='Center')

ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.legend()

# Set equal aspect
max_range = 0.06
ax1.set_xlim(center_x - max_range, center_x + max_range)
ax1.set_ylim(y_min - 0.01, y_min + 0.12)
ax1.set_zlim(center_z - max_range, center_z + max_range)

# View 2: Close-up of mounting surface
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.set_title(f'Mounting Surface Close-up\n({width:.1f}mm x {height:.1f}mm)', fontsize=10)

mount_mesh2 = Poly3DCollection(mounting_surface_triangles, alpha=0.8, edgecolor='k', linewidths=0.5)
mount_mesh2.set_facecolor([1, 0.3, 0.3])
ax2.add_collection3d(mount_mesh2)

ax2.scatter([center_x], [center_y], [center_z], c='blue', s=200, marker='*', label='Lens Hole Center')

# Draw normal vector
normal_scale = 0.03
ax2.quiver(center_x, center_y, center_z,
           avg_normal[0]*normal_scale, avg_normal[1]*normal_scale, avg_normal[2]*normal_scale,
           color='green', arrow_length_ratio=0.3, linewidth=3, label='Surface Normal')

ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_zlabel('Z (m)')
ax2.legend()

margin = 0.02
ax2.set_xlim(x_min - margin, x_max + margin)
ax2.set_ylim(y_min_surf - margin, y_max_surf + margin)
ax2.set_zlim(z_min - margin, z_max + margin)

# View 3: Top-down view (looking at square face-on)
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title('Top-Down View of Square\n(Looking at mounting surface)', fontsize=10)

for tri in mounting_surface_triangles:
    triangle = np.vstack([tri, tri[0]])
    ax3.plot(triangle[:, 0]*1000, triangle[:, 2]*1000, 'r-', linewidth=0.5, alpha=0.5)

ax3.scatter([center_x*1000], [center_z*1000], c='blue', s=200, marker='*', label='Center', zorder=10)

# Draw square boundary
ax3.plot([x_min*1000, x_max*1000, x_max*1000, x_min*1000, x_min*1000],
         [z_min*1000, z_min*1000, z_max*1000, z_max*1000, z_min*1000],
         'b--', linewidth=2, label='Bounds')

ax3.set_xlabel('X (mm)')
ax3.set_ylabel('Z (mm)')
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)
ax3.legend()

# View 4: Info text
ax4 = fig.add_subplot(2, 2, 4)
ax4.axis('off')

info_text = f"""
MOUNTING SURFACE ANALYSIS (v2)

Method: Normal clustering at tip

Found: {len(mounting_surface_triangles)} triangles

Dimensions:
  Width (X):  {width:.2f} mm
  Height (Z): {height:.2f} mm
  Shape: {"SQUARE ✓" if abs(width-height) < 3 else f"Rectangle ({width/height:.2f}:1)"}

Center Position (STL frame):
  X: {center_x:.4f} m = {center_x*1000:.2f} mm
  Y: {center_y:.4f} m = {center_y*1000:.2f} mm
  Z: {center_z:.4f} m = {center_z*1000:.2f} mm

Surface Normal:
  Direction: [{avg_normal[0]:.3f}, {avg_normal[1]:.3f}, {avg_normal[2]:.3f}]
  Tilt from Y-axis: {tilt_angle:.2f}°

Expected Features:
  - 4 screw holes at corners
  - 1 dodecagon lens hole at center
  - 32x32mm UVC camera module size

Transform to Gripper Frame:
  Mount pos: [0, -0.000218, 0.000950]
  Rotation: 180° around Y-axis
  Camera pos: [{center_x:.4f}, {-center_y-0.000218:.4f}, {-center_z+0.000950:.4f}]
"""

ax4.text(0.05, 0.5, info_text, fontsize=9, family='monospace',
         verticalalignment='center', transform=ax4.transAxes)

plt.tight_layout()

output_file = "mounting_surface_visualization_v2.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✅ Saved: {output_file}")
print("="*80)
