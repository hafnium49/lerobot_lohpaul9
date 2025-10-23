#!/usr/bin/env python3
"""
Find and visualize the square mounting surface at the tip of camera mount.
The surface should have 5 holes: 4 screw holes at corners + 1 dodecagon lens hole in center.

v3 Strategy: Look for the FRONT FACE only - triangles at the maximum Y position with +Y normals
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
print("Finding and Visualizing Square Mounting Surface (v3)")
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
y_min_global = all_vertices[:, 1].min()
y_max_global = all_vertices[:, 1].max()

print(f"\nGlobal Y range: [{y_min_global:.4f}, {y_max_global:.4f}] m")
print(f"Tip at: Y = {y_min_global:.4f} m")

# Strategy:
# 1. Find all triangles at the tip region with +Y normals
# 2. Among those, find the FRONT FACE - triangles at maximum Y position
# 3. This should be the flat mounting surface

tip_threshold = y_min_global + 0.030  # Within 30mm of tip
print(f"\nStep 1: Finding triangles at tip (Y < {tip_threshold:.4f} m) with +Y normals...")

tip_triangles = []
tip_normals = []

for i, (tri_verts, normal) in enumerate(zip(vertices, normals)):
    tri_y_min = tri_verts[:, 1].min()
    if tri_y_min < tip_threshold:
        norm = normal / np.linalg.norm(normal)
        # Look for primarily +Y normals (Y component > 0.6)
        if norm[1] > 0.6:
            tip_triangles.append(tri_verts)
            tip_normals.append(norm)

print(f"  Found {len(tip_triangles)} candidate triangles")

if len(tip_triangles) == 0:
    print("❌ No triangles found!")
    exit(1)

tip_triangles = np.array(tip_triangles)
tip_normals = np.array(tip_normals)

# Step 2: Find the front face - triangles at the maximum Y among these candidates
all_tip_verts = tip_triangles.reshape(-1, 3)
y_max_tip = all_tip_verts[:, 1].max()

print(f"\nStep 2: Front face at Y = {y_max_tip:.4f} m")
print(f"  Filtering for triangles with vertices near this Y...")

# Keep triangles that have at least one vertex very close to y_max_tip
front_face_triangles = []
front_face_normals = []

y_tolerance = 0.003  # 3mm tolerance to capture more of the surface
for tri_verts, norm in zip(tip_triangles, tip_normals):
    tri_y_max = tri_verts[:, 1].max()
    if tri_y_max > y_max_tip - y_tolerance:
        front_face_triangles.append(tri_verts)
        front_face_normals.append(norm)

print(f"  Found {len(front_face_triangles)} triangles on front face")

if len(front_face_triangles) == 0:
    print("❌ No front face triangles found!")
    exit(1)

# For analysis: use ALL triangles on the front face
# The mounting surface might have holes (4 screw holes + 1 lens hole) which fragments it
print(f"\nStep 3: Using ALL {len(front_face_triangles)} triangles on front face (includes holes)")

mounting_surface_triangles = front_face_triangles
mounting_surface_normals = front_face_normals

print(f"\nFinal: {len(mounting_surface_triangles)} triangles in mounting surface")

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
print(f"  Y: [{y_min_surf:.4f}, {y_max_surf:.4f}] m (depth: {(y_max_surf-y_min_surf)*1000:.1f} mm)")
print(f"  Z: [{z_min:.4f}, {z_max:.4f}] m (height: {height:.1f} mm)")

# Check if square (~32x32mm)
if 30 < width < 36 and 30 < height < 36 and abs(width - height) < 5:
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

# Transform to gripper frame
mount_pos = np.array([0.0, -0.000218214, 0.000949706])
camera_stl = np.array([center_x, center_y, center_z])
camera_rotated = np.array([camera_stl[0], -camera_stl[1], -camera_stl[2]])
camera_gripper = mount_pos + camera_rotated

print(f"\nCamera position in gripper frame:")
print(f"  pos=\"{camera_gripper[0]:.4f} {camera_gripper[1]:.4f} {camera_gripper[2]:.4f}\"")

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
ax1.set_ylim(y_min_global - 0.01, y_min_global + 0.12)
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
MOUNTING SURFACE ANALYSIS (v3)

Method: Front face at maximum Y

Found: {len(mounting_surface_triangles)} triangles

Dimensions:
  Width (X):  {width:.2f} mm
  Height (Z): {height:.2f} mm
  Shape: {"SQUARE ✓" if abs(width-height) < 5 else f"Rectangle ({width/height:.2f}:1)"}

Center Position (STL frame):
  X: {center_x:.4f} m = {center_x*1000:.2f} mm
  Y: {center_y:.4f} m = {center_y*1000:.2f} mm
  Z: {center_z:.4f} m = {center_z*1000:.2f} mm

Surface Normal:
  Direction: [{avg_normal[0]:.3f}, {avg_normal[1]:.3f}, {avg_normal[2]:.3f}]
  Tilt from Y-axis: {tilt_angle:.2f}°

Camera Position (Gripper Frame):
  pos="{camera_gripper[0]:.4f} {camera_gripper[1]:.4f} {camera_gripper[2]:.4f}"

Expected Features:
  - 4 screw holes at corners
  - 1 dodecagon lens hole at center
  - 32x32mm UVC camera module size
"""

ax4.text(0.05, 0.5, info_text, fontsize=9, family='monospace',
         verticalalignment='center', transform=ax4.transAxes)

plt.tight_layout()

output_file = "mounting_surface_visualization_v3.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✅ Saved: {output_file}")
print("="*80)
