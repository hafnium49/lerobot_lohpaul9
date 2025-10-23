#!/usr/bin/env python3
"""
Visualize the entire camera mount (fixed jaw) with screw holes highlighted.
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
print("Visualizing Entire Camera Mount (Fixed Jaw)")
print("="*80)

# Read STL
stl_file = "src/lerobot/envs/so101_assets/official_model/assets/Wrist_Cam_Mount_32x32_UVC_Module_SO101_MIRRORED.stl"
print(f"\nReading: {stl_file}")
normals, vertices = read_stl_binary(stl_file)

# Convert mm to meters
vertices = vertices * 0.001

print(f"Total triangles: {len(vertices)}")

# Calculate bounds
all_vertices = vertices.reshape(-1, 3)
x_min, x_max = all_vertices[:, 0].min(), all_vertices[:, 0].max()
y_min, y_max = all_vertices[:, 1].min(), all_vertices[:, 1].max()
z_min, z_max = all_vertices[:, 2].min(), all_vertices[:, 2].max()

print(f"\nCamera Mount Dimensions:")
print(f"  X: [{x_min:.4f}, {x_max:.4f}] m (width: {(x_max-x_min)*1000:.1f} mm)")
print(f"  Y: [{y_min:.4f}, {y_max:.4f}] m (length: {(y_max-y_min)*1000:.1f} mm)")
print(f"  Z: [{z_min:.4f}, {z_max:.4f}] m (height: {(z_max-z_min)*1000:.1f} mm)")

# The 4 corner screw holes
screw_holes_stl = np.array([
    [-0.0124, -0.0611, -0.0107],  # H1 (bottom-left)
    [-0.0131, -0.0858,  0.0019],  # H4 (top-left)
    [ 0.0172, -0.0616, -0.0110],  # H6 (bottom-right)
    [ 0.0181, -0.0859,  0.0019],  # H9 (top-right)
])

labels = ["H1", "H4", "H6", "H9"]

# Camera position
camera_center = np.array([0.0024, -0.0611, -0.0045])

print(f"\nCamera position (STL frame): {camera_center}")

# Identify different regions of the mount
print(f"\nAnalyzing mount geometry...")

# Classify triangles by region
base_triangles = []  # Base mounting to gripper
extension_triangles = []  # Extension arm
tip_triangles = []  # Tip with mounting surface and holes

for tri_verts in vertices:
    tri_y_center = tri_verts[:, 1].mean()

    if tri_y_center > -0.03:  # Near gripper (Y > -30mm)
        base_triangles.append(tri_verts)
    elif tri_y_center < -0.06:  # At tip (Y < -60mm)
        tip_triangles.append(tri_verts)
    else:  # Middle extension
        extension_triangles.append(tri_verts)

base_triangles = np.array(base_triangles) if base_triangles else np.empty((0, 3, 3))
extension_triangles = np.array(extension_triangles) if extension_triangles else np.empty((0, 3, 3))
tip_triangles = np.array(tip_triangles) if tip_triangles else np.empty((0, 3, 3))

print(f"  Base: {len(base_triangles)} triangles")
print(f"  Extension: {len(extension_triangles)} triangles")
print(f"  Tip: {len(tip_triangles)} triangles")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 16))

# View 1: Full mount - 3D perspective
ax1 = fig.add_subplot(3, 3, 1, projection='3d')
ax1.set_title('Full Camera Mount - 3D View', fontsize=12, fontweight='bold')

# Plot different regions in different colors
if len(base_triangles) > 0:
    base_mesh = Poly3DCollection(base_triangles, alpha=0.5, edgecolor='k', linewidths=0.1)
    base_mesh.set_facecolor([0.7, 0.9, 0.7])  # Light green
    ax1.add_collection3d(base_mesh)

if len(extension_triangles) > 0:
    ext_mesh = Poly3DCollection(extension_triangles, alpha=0.5, edgecolor='k', linewidths=0.1)
    ext_mesh.set_facecolor([0.7, 0.7, 0.9])  # Light blue
    ax1.add_collection3d(ext_mesh)

if len(tip_triangles) > 0:
    tip_mesh = Poly3DCollection(tip_triangles, alpha=0.5, edgecolor='k', linewidths=0.1)
    tip_mesh.set_facecolor([0.9, 0.7, 0.7])  # Light red
    ax1.add_collection3d(tip_mesh)

# Plot screw holes
colors = ['red', 'green', 'blue', 'orange']
for hole, label, color in zip(screw_holes_stl, labels, colors):
    ax1.scatter([hole[0]], [hole[1]], [hole[2]], c=color, s=200, marker='o',
                edgecolors='black', linewidths=2, zorder=10, label=label)

# Plot camera
ax1.scatter([camera_center[0]], [camera_center[1]], [camera_center[2]],
            c='purple', s=400, marker='*', edgecolors='black', linewidths=2, zorder=15,
            label='Camera')

ax1.set_xlabel('X (m)', fontweight='bold')
ax1.set_ylabel('Y (m)', fontweight='bold')
ax1.set_zlabel('Z (m)', fontweight='bold')
ax1.legend(loc='upper left', fontsize=8, ncol=2)
ax1.view_init(elev=20, azim=45)

# Set limits
ax1.set_xlim(x_min - 0.01, x_max + 0.01)
ax1.set_ylim(y_min - 0.01, y_max + 0.01)
ax1.set_zlim(z_min - 0.01, z_max + 0.01)

# View 2: Full mount - Side view
ax2 = fig.add_subplot(3, 3, 2, projection='3d')
ax2.set_title('Full Camera Mount - Side View', fontsize=12, fontweight='bold')

# Plot all triangles (subsampled for clarity)
full_mesh = Poly3DCollection(vertices[::5], alpha=0.4, edgecolor='k', linewidths=0.2)
full_mesh.set_facecolor([0.6, 0.6, 0.6])
ax2.add_collection3d(full_mesh)

# Highlight screw holes and camera
for hole, label, color in zip(screw_holes_stl, labels, colors):
    ax2.scatter([hole[0]], [hole[1]], [hole[2]], c=color, s=200, marker='o',
                edgecolors='black', linewidths=2, zorder=10)

ax2.scatter([camera_center[0]], [camera_center[1]], [camera_center[2]],
            c='purple', s=400, marker='*', edgecolors='black', linewidths=2, zorder=15)

ax2.set_xlabel('X (m)', fontweight='bold')
ax2.set_ylabel('Y (m)', fontweight='bold')
ax2.set_zlabel('Z (m)', fontweight='bold')
ax2.view_init(elev=0, azim=0)  # Side view

ax2.set_xlim(x_min - 0.01, x_max + 0.01)
ax2.set_ylim(y_min - 0.01, y_max + 0.01)
ax2.set_zlim(z_min - 0.01, z_max + 0.01)

# View 3: Full mount - Top view
ax3 = fig.add_subplot(3, 3, 3, projection='3d')
ax3.set_title('Full Camera Mount - Top View', fontsize=12, fontweight='bold')

# Plot all triangles
full_mesh = Poly3DCollection(vertices[::5], alpha=0.4, edgecolor='k', linewidths=0.2)
full_mesh.set_facecolor([0.6, 0.6, 0.6])
ax3.add_collection3d(full_mesh)

# Highlight screw holes and camera
for hole, label, color in zip(screw_holes_stl, labels, colors):
    ax3.scatter([hole[0]], [hole[1]], [hole[2]], c=color, s=200, marker='o',
                edgecolors='black', linewidths=2, zorder=10)

ax3.scatter([camera_center[0]], [camera_center[1]], [camera_center[2]],
            c='purple', s=400, marker='*', edgecolors='black', linewidths=2, zorder=15)

ax3.set_xlabel('X (m)', fontweight='bold')
ax3.set_ylabel('Y (m)', fontweight='bold')
ax3.set_zlabel('Z (m)', fontweight='bold')
ax3.view_init(elev=90, azim=-90)  # Top view

ax3.set_xlim(x_min - 0.01, x_max + 0.01)
ax3.set_ylim(y_min - 0.01, y_max + 0.01)
ax3.set_zlim(z_min - 0.01, z_max + 0.01)

# View 4: XY plane projection
ax4 = fig.add_subplot(3, 3, 4)
ax4.set_title('Side Profile (XY plane)', fontsize=12, fontweight='bold')

# Plot edges
for tri in vertices[::20]:
    triangle = np.vstack([tri, tri[0]])
    ax4.plot(triangle[:, 0]*1000, triangle[:, 1]*1000, 'gray', linewidth=0.3, alpha=0.3)

# Plot screw holes
for hole, label, color in zip(screw_holes_stl, labels, colors):
    ax4.scatter(hole[0]*1000, hole[1]*1000, c=color, s=200, marker='o',
                edgecolors='black', linewidths=2, zorder=10, label=label)

# Plot camera
ax4.scatter(camera_center[0]*1000, camera_center[1]*1000, c='purple', s=400,
            marker='*', edgecolors='black', linewidths=2, zorder=15, label='Camera')

# Add region labels
ax4.axhline(-30, color='green', linestyle='--', alpha=0.5, linewidth=2)
ax4.axhline(-60, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax4.text(25, -15, 'BASE', fontsize=10, fontweight='bold', color='green')
ax4.text(25, -45, 'EXTENSION', fontsize=10, fontweight='bold', color='blue')
ax4.text(25, -75, 'TIP', fontsize=10, fontweight='bold', color='red')

ax4.set_xlabel('X (mm)', fontweight='bold')
ax4.set_ylabel('Y (mm)', fontweight='bold')
ax4.legend(loc='lower right', fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.set_aspect('equal')

# View 5: YZ plane projection
ax5 = fig.add_subplot(3, 3, 5)
ax5.set_title('Front Profile (YZ plane)', fontsize=12, fontweight='bold')

# Plot edges
for tri in vertices[::20]:
    triangle = np.vstack([tri, tri[0]])
    ax5.plot(triangle[:, 1]*1000, triangle[:, 2]*1000, 'gray', linewidth=0.3, alpha=0.3)

# Plot screw holes
for hole, label, color in zip(screw_holes_stl, labels, colors):
    ax5.scatter(hole[1]*1000, hole[2]*1000, c=color, s=200, marker='o',
                edgecolors='black', linewidths=2, zorder=10)

# Plot camera
ax5.scatter(camera_center[1]*1000, camera_center[2]*1000, c='purple', s=400,
            marker='*', edgecolors='black', linewidths=2, zorder=15)

ax5.set_xlabel('Y (mm)', fontweight='bold')
ax5.set_ylabel('Z (mm)', fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.set_aspect('equal')

# View 6: XZ plane projection (top-down on mount)
ax6 = fig.add_subplot(3, 3, 6)
ax6.set_title('Top-Down Profile (XZ plane)', fontsize=12, fontweight='bold')

# Plot edges
for tri in vertices[::20]:
    triangle = np.vstack([tri, tri[0]])
    ax6.plot(triangle[:, 0]*1000, triangle[:, 2]*1000, 'gray', linewidth=0.3, alpha=0.3)

# Plot screw holes
for hole, label, color in zip(screw_holes_stl, labels, colors):
    ax6.scatter(hole[0]*1000, hole[2]*1000, c=color, s=200, marker='o',
                edgecolors='black', linewidths=2, zorder=10, label=label)

# Plot camera
ax6.scatter(camera_center[0]*1000, camera_center[2]*1000, c='purple', s=400,
            marker='*', edgecolors='black', linewidths=2, zorder=15, label='Camera')

ax6.set_xlabel('X (mm)', fontweight='bold')
ax6.set_ylabel('Z (mm)', fontweight='bold')
ax6.legend(loc='upper right', fontsize=8)
ax6.grid(True, alpha=0.3)
ax6.set_aspect('equal')

# View 7: Close-up of tip with mounting surface
ax7 = fig.add_subplot(3, 3, 7, projection='3d')
ax7.set_title('Close-up: Mounting Surface at Tip', fontsize=12, fontweight='bold')

# Plot only tip triangles
if len(tip_triangles) > 0:
    tip_mesh_close = Poly3DCollection(tip_triangles, alpha=0.6, edgecolor='k', linewidths=0.3)
    tip_mesh_close.set_facecolor([0.9, 0.7, 0.7])
    ax7.add_collection3d(tip_mesh_close)

# Plot screw holes - larger
for hole, label, color in zip(screw_holes_stl, labels, colors):
    ax7.scatter([hole[0]], [hole[1]], [hole[2]], c=color, s=300, marker='o',
                edgecolors='black', linewidths=2, zorder=10, label=label)

# Plot camera - larger
ax7.scatter([camera_center[0]], [camera_center[1]], [camera_center[2]],
            c='purple', s=600, marker='*', edgecolors='black', linewidths=3, zorder=15,
            label='Camera')

# Draw lines connecting screw holes
hole_order = [0, 1, 3, 2, 0]
for i in range(len(hole_order)-1):
    h1, h2 = hole_order[i], hole_order[i+1]
    ax7.plot([screw_holes_stl[h1,0], screw_holes_stl[h2,0]],
             [screw_holes_stl[h1,1], screw_holes_stl[h2,1]],
             [screw_holes_stl[h1,2], screw_holes_stl[h2,2]],
             'k--', linewidth=2, alpha=0.7)

ax7.set_xlabel('X (m)', fontweight='bold')
ax7.set_ylabel('Y (m)', fontweight='bold')
ax7.set_zlabel('Z (m)', fontweight='bold')
ax7.legend(loc='upper left', fontsize=7)
ax7.view_init(elev=30, azim=45)

# Zoom to tip area
margin = 0.015
ax7.set_xlim(camera_center[0] - margin, camera_center[0] + margin)
ax7.set_ylim(camera_center[1] - margin, camera_center[1] + margin)
ax7.set_zlim(camera_center[2] - margin, camera_center[2] + margin)

# View 8: Wireframe view
ax8 = fig.add_subplot(3, 3, 8, projection='3d')
ax8.set_title('Wireframe View', fontsize=12, fontweight='bold')

# Plot as wireframe
for tri in vertices[::10]:
    triangle = np.vstack([tri, tri[0]])
    ax8.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2],
             'gray', linewidth=0.3, alpha=0.4)

# Highlight screw holes and camera
for hole, color in zip(screw_holes_stl, colors):
    ax8.scatter([hole[0]], [hole[1]], [hole[2]], c=color, s=200, marker='o',
                edgecolors='black', linewidths=2, zorder=10)

ax8.scatter([camera_center[0]], [camera_center[1]], [camera_center[2]],
            c='purple', s=400, marker='*', edgecolors='black', linewidths=2, zorder=15)

ax8.set_xlabel('X (m)', fontweight='bold')
ax8.set_ylabel('Y (m)', fontweight='bold')
ax8.set_zlabel('Z (m)', fontweight='bold')
ax8.view_init(elev=20, azim=-60)

ax8.set_xlim(x_min - 0.01, x_max + 0.01)
ax8.set_ylim(y_min - 0.01, y_max + 0.01)
ax8.set_zlim(z_min - 0.01, z_max + 0.01)

# View 9: Summary info
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis('off')

info_text = f"""
CAMERA MOUNT GEOMETRY SUMMARY

Overall Dimensions:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Width (X):  {(x_max-x_min)*1000:.1f} mm
  Length (Y): {(y_max-y_min)*1000:.1f} mm
  Height (Z): {(z_max-z_min)*1000:.1f} mm

Total Triangles: {len(vertices)}

Mount Regions:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Base (gripper mount):    {len(base_triangles):5d} triangles
  Extension (arm):         {len(extension_triangles):5d} triangles
  Tip (mounting surface):  {len(tip_triangles):5d} triangles

4 Corner Screw Holes:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  H1 (red):    [{screw_holes_stl[0,0]*1000:5.1f}, {screw_holes_stl[0,1]*1000:6.1f}, {screw_holes_stl[0,2]*1000:6.1f}] mm
  H4 (green):  [{screw_holes_stl[1,0]*1000:5.1f}, {screw_holes_stl[1,1]*1000:6.1f}, {screw_holes_stl[1,2]*1000:6.1f}] mm
  H6 (blue):   [{screw_holes_stl[2,0]*1000:5.1f}, {screw_holes_stl[2,1]*1000:6.1f}, {screw_holes_stl[2,2]*1000:6.1f}] mm
  H9 (orange): [{screw_holes_stl[3,0]*1000:5.1f}, {screw_holes_stl[3,1]*1000:6.1f}, {screw_holes_stl[3,2]*1000:6.1f}] mm

Camera Position (purple star):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STL frame: [{camera_center[0]:.4f}, {camera_center[1]:.4f}, {camera_center[2]:.4f}] m
             [{camera_center[0]*1000:.1f}, {camera_center[1]*1000:.1f}, {camera_center[2]*1000:.1f}] mm

Notes:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Camera at geometric center of 4 screw holes
• Mounting surface at tip (Y < -60mm)
• STL coordinates: Y-axis points toward gripper
• In MuJoCo: 180° rotation around Y-axis
"""

ax9.text(0.05, 0.5, info_text, fontsize=9, family='monospace',
         verticalalignment='center', transform=ax9.transAxes,
         bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow',
                   alpha=0.8, edgecolor='black', linewidth=2))

plt.suptitle('Complete Camera Mount (Fixed Jaw) Visualization',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])

output_file = "complete_camera_mount_visualization.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✅ Saved: {output_file}")
print("="*80)
