#!/usr/bin/env python3
"""
Extract and visualize the central dodecagon lens hole from the mounting surface.
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
print("Extracting Central Dodecagon Lens Hole")
print("="*80)

# Read STL
stl_file = "src/lerobot/envs/so101_assets/official_model/assets/Wrist_Cam_Mount_32x32_UVC_Module_SO101_MIRRORED.stl"
print(f"\nReading: {stl_file}")
normals, vertices = read_stl_binary(stl_file)

# Convert mm to meters
vertices = vertices * 0.001

print(f"Total triangles: {len(vertices)}")

# Strategy: Find the central opening by looking for triangles that:
# 1. Are at the tip (Y < -0.06)
# 2. Have normals that are NOT pointing outward (+Y)
# 3. Are near the center in XZ plane (within ~10mm radius)
# 4. Form the inner walls of the lens hole

all_vertices = vertices.reshape(-1, 3)
y_min = all_vertices[:, 1].min()

# Camera center position (from previous analysis)
camera_center = np.array([0.0024, -0.0611, -0.0045])

print(f"\nCamera center: [{camera_center[0]:.4f}, {camera_center[1]:.4f}, {camera_center[2]:.4f}] m")

# Step 1: Find triangles at the tip near the mounting surface
tip_threshold = y_min + 0.030  # Within 30mm of tip
print(f"\nStep 1: Finding triangles at tip (Y < {tip_threshold:.4f} m)...")

tip_triangles = []
tip_normals = []

for tri_verts, normal in zip(vertices, normals):
    tri_y_min = tri_verts[:, 1].min()
    if tri_y_min < tip_threshold:
        tip_triangles.append(tri_verts)
        tip_normals.append(normal / np.linalg.norm(normal))

tip_triangles = np.array(tip_triangles)
tip_normals = np.array(tip_normals)

print(f"  Found {len(tip_triangles)} triangles at tip")

# Step 2: Find triangles near the camera center in XZ plane
print(f"\nStep 2: Finding triangles near camera center...")

radius_threshold = 0.012  # Within 12mm of center in XZ plane
central_triangles = []
central_normals = []

for tri_verts, norm in zip(tip_triangles, tip_normals):
    # Calculate triangle center
    tri_center = tri_verts.mean(axis=0)

    # Distance from camera center in XZ plane only
    dx = tri_center[0] - camera_center[0]
    dz = tri_center[2] - camera_center[2]
    dist_xz = np.sqrt(dx**2 + dz**2)

    if dist_xz < radius_threshold:
        central_triangles.append(tri_verts)
        central_normals.append(norm)

central_triangles = np.array(central_triangles)
central_normals = np.array(central_normals)

print(f"  Found {len(central_triangles)} triangles near center")

# Step 3: Separate into hole walls vs mounting surface
print(f"\nStep 3: Separating hole walls from mounting surface...")

# Hole walls should have normals pointing radially (small Y component)
# Mounting surface has normals pointing in +Y direction

hole_wall_triangles = []
hole_wall_normals = []
surface_triangles = []
surface_normals = []

for tri_verts, norm in zip(central_triangles, central_normals):
    # Check Y component of normal
    if abs(norm[1]) < 0.6:  # Radial normals (hole walls)
        hole_wall_triangles.append(tri_verts)
        hole_wall_normals.append(norm)
    else:  # Axial normals (mounting surface)
        surface_triangles.append(tri_verts)
        surface_normals.append(norm)

hole_wall_triangles = np.array(hole_wall_triangles) if hole_wall_triangles else np.empty((0, 3, 3))
hole_wall_normals = np.array(hole_wall_normals) if hole_wall_normals else np.empty((0, 3))
surface_triangles = np.array(surface_triangles) if surface_triangles else np.empty((0, 3, 3))
surface_normals = np.array(surface_normals) if surface_normals else np.empty((0, 3))

print(f"  Hole wall triangles: {len(hole_wall_triangles)}")
print(f"  Surface triangles: {len(surface_triangles)}")

# Step 4: Analyze hole geometry
if len(hole_wall_triangles) > 0:
    print(f"\nStep 4: Analyzing dodecagon hole geometry...")

    hole_verts = hole_wall_triangles.reshape(-1, 3)

    # Find unique vertices on the hole edge (at the surface Y position)
    y_surface = -0.0611  # Known surface Y position
    y_tolerance = 0.003  # 3mm tolerance

    edge_vertices = []
    for v in hole_verts:
        if abs(v[1] - y_surface) < y_tolerance:
            # Check if this vertex is new
            is_new = True
            for ev in edge_vertices:
                if np.linalg.norm(v - ev) < 0.001:  # Within 1mm
                    is_new = False
                    break
            if is_new:
                edge_vertices.append(v)

    edge_vertices = np.array(edge_vertices)
    print(f"  Found {len(edge_vertices)} edge vertices at surface")

    # Analyze hole shape
    if len(edge_vertices) > 0:
        # Calculate hole center
        hole_center_x = edge_vertices[:, 0].mean()
        hole_center_z = edge_vertices[:, 2].mean()

        print(f"  Hole center (XZ): [{hole_center_x*1000:.1f}, {hole_center_z*1000:.1f}] mm")

        # Calculate radial distances
        radii = []
        for v in edge_vertices:
            dx = v[0] - hole_center_x
            dz = v[2] - hole_center_z
            r = np.sqrt(dx**2 + dz**2)
            radii.append(r)

        radii = np.array(radii)
        print(f"  Hole radius: min={radii.min()*1000:.1f} mm, max={radii.max()*1000:.1f} mm, mean={radii.mean()*1000:.1f} mm")
        print(f"  Hole diameter: ~{radii.mean()*2*1000:.1f} mm")

        # Try to identify dodecagon (12-sided) shape by analyzing angular distribution
        angles = []
        for v in edge_vertices:
            dx = v[0] - hole_center_x
            dz = v[2] - hole_center_z
            angle = np.arctan2(dz, dx)
            angles.append(angle)

        angles = np.array(angles)
        angles_deg = np.degrees(angles)

        # Sort angles
        sorted_angles = np.sort(angles_deg)

        print(f"\n  Angular distribution of edge vertices:")
        print(f"    Number of vertices: {len(sorted_angles)}")
        print(f"    Angles (degrees): {sorted_angles}")

        # Check if vertices form regular dodecagon (12 vertices at 30° intervals)
        mean_diff = None
        if len(sorted_angles) >= 2:
            angular_diffs = np.diff(sorted_angles)
            mean_diff = angular_diffs.mean()
            print(f"    Mean angular spacing: {mean_diff:.1f}°")
            print(f"    Expected for dodecagon: 30.0°")

            if 25 < mean_diff < 35 and len(sorted_angles) >= 10:
                print(f"    ✅ Shape is consistent with dodecagon!")
            else:
                print(f"    ⚠️  Shape may not be a regular dodecagon (found {len(sorted_angles)} vertices)")

# Create visualization
print("\nGenerating visualization...")
fig = plt.figure(figsize=(20, 14))

# View 1: 3D view of hole
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.set_title('Dodecagon Lens Hole - 3D View', fontsize=12, fontweight='bold')

if len(hole_wall_triangles) > 0:
    hole_mesh = Poly3DCollection(hole_wall_triangles, alpha=0.7, edgecolor='darkred', linewidths=0.5)
    hole_mesh.set_facecolor([1, 0.3, 0.3])
    ax1.add_collection3d(hole_mesh)

if len(surface_triangles) > 0:
    surf_mesh = Poly3DCollection(surface_triangles, alpha=0.3, edgecolor='k', linewidths=0.3)
    surf_mesh.set_facecolor([0.7, 0.7, 0.7])
    ax1.add_collection3d(surf_mesh)

# Plot camera center
ax1.scatter([camera_center[0]], [camera_center[1]], [camera_center[2]],
            c='blue', s=300, marker='*', edgecolors='black', linewidths=2, label='Camera Center')

ax1.set_xlabel('X (m)', fontweight='bold')
ax1.set_ylabel('Y (m)', fontweight='bold')
ax1.set_zlabel('Z (m)', fontweight='bold')
ax1.legend()
ax1.view_init(elev=20, azim=45)

margin = 0.015
ax1.set_xlim(camera_center[0] - margin, camera_center[0] + margin)
ax1.set_ylim(camera_center[1] - margin, camera_center[1] + margin)
ax1.set_zlim(camera_center[2] - margin, camera_center[2] + margin)

# View 2: Top-down view (XZ plane) - hole opening
ax2 = fig.add_subplot(2, 3, 2)
ax2.set_title('Hole Opening - Top View (XZ plane)', fontsize=12, fontweight='bold')

# Plot hole wall triangles from above
if len(hole_wall_triangles) > 0:
    for tri in hole_wall_triangles:
        triangle = np.vstack([tri, tri[0]])
        ax2.plot(triangle[:, 0]*1000, triangle[:, 2]*1000, 'r-', linewidth=0.5, alpha=0.5)

# Plot edge vertices
if len(edge_vertices) > 0:
    ax2.scatter(edge_vertices[:, 0]*1000, edge_vertices[:, 2]*1000,
                c='red', s=100, marker='o', edgecolors='black', linewidths=1.5, zorder=10,
                label=f'Edge vertices ({len(edge_vertices)})')

    # Connect edge vertices to show shape
    if len(edge_vertices) > 2:
        # Sort by angle
        hole_center_xz = np.array([hole_center_x, hole_center_z])
        angles_for_sort = []
        for v in edge_vertices:
            dx = v[0] - hole_center_x
            dz = v[2] - hole_center_z
            angle = np.arctan2(dz, dx)
            angles_for_sort.append(angle)

        sorted_indices = np.argsort(angles_for_sort)
        sorted_verts = edge_vertices[sorted_indices]

        # Draw polygon
        polygon_x = np.append(sorted_verts[:, 0], sorted_verts[0, 0]) * 1000
        polygon_z = np.append(sorted_verts[:, 2], sorted_verts[0, 2]) * 1000
        ax2.plot(polygon_x, polygon_z, 'b-', linewidth=2, label='Hole edge')
        ax2.fill(polygon_x, polygon_z, color='lightblue', alpha=0.3)

# Plot camera center
ax2.scatter([camera_center[0]*1000], [camera_center[2]*1000],
            c='blue', s=400, marker='*', edgecolors='black', linewidths=2,
            zorder=15, label='Camera Center')

ax2.set_xlabel('X (mm)', fontweight='bold')
ax2.set_ylabel('Z (mm)', fontweight='bold')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.legend()

# View 3: Side view showing hole depth
ax3 = fig.add_subplot(2, 3, 3)
ax3.set_title('Hole Depth - Side View (XY plane)', fontsize=12, fontweight='bold')

if len(hole_wall_triangles) > 0:
    for tri in hole_wall_triangles:
        triangle = np.vstack([tri, tri[0]])
        ax3.plot(triangle[:, 0]*1000, triangle[:, 1]*1000, 'r-', linewidth=0.5, alpha=0.5)

ax3.scatter([camera_center[0]*1000], [camera_center[1]*1000],
            c='blue', s=400, marker='*', edgecolors='black', linewidths=2)

ax3.set_xlabel('X (mm)', fontweight='bold')
ax3.set_ylabel('Y (mm)', fontweight='bold')
ax3.grid(True, alpha=0.3)

# View 4: Detailed hole geometry
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
ax4.set_title('Hole Wall Geometry - Detail', fontsize=12, fontweight='bold')

if len(hole_wall_triangles) > 0:
    hole_mesh2 = Poly3DCollection(hole_wall_triangles, alpha=0.8, edgecolor='black', linewidths=0.5)
    hole_mesh2.set_facecolor([1, 0.4, 0.4])
    ax4.add_collection3d(hole_mesh2)

ax4.set_xlabel('X (m)')
ax4.set_ylabel('Y (m)')
ax4.set_zlabel('Z (m)')
ax4.view_init(elev=30, azim=-60)

if len(hole_wall_triangles) > 0:
    hole_verts_all = hole_wall_triangles.reshape(-1, 3)
    ax4.set_xlim(hole_verts_all[:, 0].min() - 0.002, hole_verts_all[:, 0].max() + 0.002)
    ax4.set_ylim(hole_verts_all[:, 1].min() - 0.002, hole_verts_all[:, 1].max() + 0.002)
    ax4.set_zlim(hole_verts_all[:, 2].min() - 0.002, hole_verts_all[:, 2].max() + 0.002)

# View 5: Normals visualization
ax5 = fig.add_subplot(2, 3, 5)
ax5.set_title('Hole Wall Normals (XZ projection)', fontsize=12, fontweight='bold')

if len(hole_wall_triangles) > 0:
    for tri, norm in zip(hole_wall_triangles, hole_wall_normals):
        tri_center = tri.mean(axis=0)
        # Project onto XZ plane and draw normal
        ax5.arrow(tri_center[0]*1000, tri_center[2]*1000,
                 norm[0]*3, norm[2]*3,
                 head_width=0.5, head_length=0.3, fc='red', ec='red', alpha=0.6)

if len(edge_vertices) > 0:
    ax5.scatter(edge_vertices[:, 0]*1000, edge_vertices[:, 2]*1000,
                c='blue', s=100, marker='o', edgecolors='black', linewidths=1.5, zorder=10)

ax5.set_xlabel('X (mm)', fontweight='bold')
ax5.set_ylabel('Z (mm)', fontweight='bold')
ax5.set_aspect('equal')
ax5.grid(True, alpha=0.3)

# View 6: Summary info
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')

if len(edge_vertices) > 0:
    mean_spacing_str = f"{mean_diff:.1f}" if mean_diff is not None else "N/A"
    info_text = f"""
DODECAGON LENS HOLE ANALYSIS

Extraction Criteria:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • At tip: Y < {tip_threshold:.4f} m
  • Near center: XZ distance < {radius_threshold*1000:.1f} mm
  • Radial normals: |Y-component| < 0.6

Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Hole wall triangles: {len(hole_wall_triangles)}
  Surface triangles: {len(surface_triangles)}
  Edge vertices: {len(edge_vertices)}

Hole Geometry:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Center (XZ): [{hole_center_x*1000:.1f}, {hole_center_z*1000:.1f}] mm
  Radius: {radii.mean()*1000:.1f} mm (±{radii.std()*1000:.1f} mm)
  Diameter: {radii.mean()*2*1000:.1f} mm

Angular Distribution:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Vertices: {len(sorted_angles)}
  Mean spacing: {mean_spacing_str}°
  Expected (dodecagon): 30.0°

Camera Position:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STL: [{camera_center[0]:.4f}, {camera_center[1]:.4f}, {camera_center[2]:.4f}] m

Notes:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Camera positioned at center of lens hole
  • Hole walls have radially-pointing normals
  • Edge vertices define dodecagon perimeter
"""
else:
    info_text = f"""
DODECAGON LENS HOLE ANALYSIS

Extraction Criteria:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • At tip: Y < {tip_threshold:.4f} m
  • Near center: XZ distance < {radius_threshold*1000:.1f} mm
  • Radial normals: |Y-component| < 0.6

Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Hole wall triangles: {len(hole_wall_triangles)}
  Surface triangles: {len(surface_triangles)}

⚠️ Insufficient edge vertices found
  Try adjusting search parameters
"""

ax6.text(0.05, 0.5, info_text, fontsize=9, family='monospace',
         verticalalignment='center', transform=ax6.transAxes,
         bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow',
                   alpha=0.8, edgecolor='black', linewidth=2))

plt.suptitle('Dodecagon Lens Hole Extraction from Camera Mount',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.97])

output_file = "dodecagon_hole_extraction.png"
plt.savefig(output_file, dpi=200, bbox_inches='tight')
print(f"\n✅ Saved: {output_file}")
print("="*80)
