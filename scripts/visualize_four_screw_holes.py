#!/usr/bin/env python3
"""
Visualize the 4 corner screw holes on the mounting surface.
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
print("Visualizing 4 Corner Screw Holes")
print("="*80)

# The 4 corner screw holes from previous analysis
screw_holes_stl = np.array([
    [-0.0124, -0.0611, -0.0107],  # H1 (bottom-left)
    [-0.0131, -0.0858,  0.0019],  # H4 (top-left)
    [ 0.0172, -0.0616, -0.0110],  # H6 (bottom-right)
    [ 0.0181, -0.0859,  0.0019],  # H9 (top-right)
])

labels = ["H1\n(bottom-left)", "H4\n(top-left)", "H6\n(bottom-right)", "H9\n(top-right)"]

print("\n4 Corner Screw Holes (STL frame):")
for hole, label in zip(screw_holes_stl, labels):
    print(f"  {label.replace(chr(10), ' '):25s}: [{hole[0]:7.4f}, {hole[1]:7.4f}, {hole[2]:7.4f}] m")

# Calculate center (camera position)
center = screw_holes_stl.mean(axis=0)
# Use most outward Y for surface
center[1] = screw_holes_stl[:, 1].max()

print(f"\nCamera Position (center at surface):")
print(f"  STL frame: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}] m")

# Transform to gripper frame
mount_pos = np.array([0.0, -0.000218214, 0.000949706])
center_rotated = np.array([center[0], -center[1], -center[2]])
center_gripper = mount_pos + center_rotated
print(f"  Gripper frame: [{center_gripper[0]:.4f}, {center_gripper[1]:.4f}, {center_gripper[2]:.4f}] m")

# Read STL for context
stl_file = "src/lerobot/envs/so101_assets/official_model/assets/Wrist_Cam_Mount_32x32_UVC_Module_SO101_MIRRORED.stl"
print(f"\nReading STL: {stl_file}")
normals, vertices = read_stl_binary(stl_file)
vertices = vertices * 0.001  # mm to m

# Find mounting surface triangles for context
all_vertices = vertices.reshape(-1, 3)
y_min = all_vertices[:, 1].min()
tip_threshold = y_min + 0.030

surface_triangles = []
for tri_verts, normal in zip(vertices, normals):
    tri_y_min = tri_verts[:, 1].min()
    if tri_y_min < tip_threshold:
        norm = normal / np.linalg.norm(normal)
        if norm[1] > 0.6:  # +Y facing
            surface_triangles.append(tri_verts)

surface_triangles = np.array(surface_triangles)
print(f"Found {len(surface_triangles)} mounting surface triangles for context")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 14))

# View 1: 3D view with camera mount
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.set_title('3D View: Camera Mount with Screw Holes', fontsize=12, fontweight='bold')

# Plot mounting surface
if len(surface_triangles) > 0:
    surf_mesh = Poly3DCollection(surface_triangles, alpha=0.3, edgecolor='k', linewidths=0.2)
    surf_mesh.set_facecolor([0.8, 0.8, 0.8])
    ax1.add_collection3d(surf_mesh)

# Plot screw holes
colors = ['red', 'green', 'blue', 'orange']
for i, (hole, label, color) in enumerate(zip(screw_holes_stl, labels, colors)):
    ax1.scatter([hole[0]], [hole[1]], [hole[2]], c=color, s=300, marker='o',
                edgecolors='black', linewidths=2, zorder=10, label=label.replace('\n', ' '))

# Plot camera center
ax1.scatter([center[0]], [center[1]], [center[2]], c='purple', s=500, marker='*',
            edgecolors='black', linewidths=2, zorder=15, label='Camera Center')

# Draw lines connecting screw holes to show square pattern
hole_order = [0, 1, 3, 2, 0]  # H1 -> H4 -> H9 -> H6 -> H1
for i in range(len(hole_order)-1):
    h1, h2 = hole_order[i], hole_order[i+1]
    ax1.plot([screw_holes_stl[h1,0], screw_holes_stl[h2,0]],
             [screw_holes_stl[h1,1], screw_holes_stl[h2,1]],
             [screw_holes_stl[h1,2], screw_holes_stl[h2,2]],
             'k--', linewidth=2, alpha=0.5)

ax1.set_xlabel('X (m)', fontweight='bold')
ax1.set_ylabel('Y (m)', fontweight='bold')
ax1.set_zlabel('Z (m)', fontweight='bold')
ax1.legend(loc='upper left', fontsize=8)
ax1.view_init(elev=20, azim=45)

# View 2: Top-down view (XZ plane - looking at mounting surface face-on)
ax2 = fig.add_subplot(2, 3, 2)
ax2.set_title('Top-Down View: Mounting Surface Face\n(Looking at surface from above)',
              fontsize=12, fontweight='bold')

# Plot surface triangles
for tri in surface_triangles:
    triangle = np.vstack([tri, tri[0]])
    ax2.plot(triangle[:, 0]*1000, triangle[:, 2]*1000, 'gray', linewidth=0.5, alpha=0.2)

# Plot screw holes
for i, (hole, label, color) in enumerate(zip(screw_holes_stl, labels, colors)):
    ax2.scatter(hole[0]*1000, hole[2]*1000, c=color, s=400, marker='o',
                edgecolors='black', linewidths=2, zorder=10)
    # Add label with offset
    offset_x = 3 if hole[0] > 0 else -3
    offset_z = 3 if hole[2] > 0 else -3
    ax2.text(hole[0]*1000 + offset_x, hole[2]*1000 + offset_z,
             label.replace('\n', ' '), fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))

# Plot camera center
ax2.scatter(center[0]*1000, center[2]*1000, c='purple', s=600, marker='*',
            edgecolors='black', linewidths=2, zorder=15)
ax2.text(center[0]*1000, center[2]*1000 - 5, 'Camera\nCenter',
         fontsize=10, ha='center', va='top', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='purple', alpha=0.3))

# Draw rectangle connecting screw holes
for i in range(len(hole_order)-1):
    h1, h2 = hole_order[i], hole_order[i+1]
    ax2.plot([screw_holes_stl[h1,0]*1000, screw_holes_stl[h2,0]*1000],
             [screw_holes_stl[h1,2]*1000, screw_holes_stl[h2,2]*1000],
             'k-', linewidth=3, alpha=0.7, zorder=5)

# Add dimensions
x_span = (screw_holes_stl[:, 0].max() - screw_holes_stl[:, 0].min()) * 1000
z_span = (screw_holes_stl[:, 2].max() - screw_holes_stl[:, 2].min()) * 1000

ax2.annotate('', xy=(screw_holes_stl[2,0]*1000, -15),
             xytext=(screw_holes_stl[0,0]*1000, -15),
             arrowprops=dict(arrowstyle='<->', lw=2, color='blue'))
ax2.text((screw_holes_stl[0,0] + screw_holes_stl[2,0])/2*1000, -17,
         f'{x_span:.1f} mm', ha='center', va='top', fontsize=11,
         fontweight='bold', color='blue')

ax2.annotate('', xy=(22, screw_holes_stl[1,2]*1000),
             xytext=(22, screw_holes_stl[0,2]*1000),
             arrowprops=dict(arrowstyle='<->', lw=2, color='red'))
ax2.text(24, (screw_holes_stl[1,2] + screw_holes_stl[0,2])/2*1000,
         f'{z_span:.1f} mm', ha='left', va='center', fontsize=11,
         fontweight='bold', color='red', rotation=90)

ax2.set_xlabel('X (mm)', fontweight='bold')
ax2.set_ylabel('Z (mm)', fontweight='bold')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-20, 25)
ax2.set_ylim(-16, 8)

# View 3: Side view (XY plane)
ax3 = fig.add_subplot(2, 3, 3)
ax3.set_title('Side View (XY plane)', fontsize=12, fontweight='bold')

for i, (hole, label, color) in enumerate(zip(screw_holes_stl, labels, colors)):
    ax3.scatter(hole[0]*1000, hole[1]*1000, c=color, s=400, marker='o',
                edgecolors='black', linewidths=2, zorder=10)

ax3.scatter(center[0]*1000, center[1]*1000, c='purple', s=600, marker='*',
            edgecolors='black', linewidths=2, zorder=15)

# Show Y-span
y_span = (screw_holes_stl[:, 1].max() - screw_holes_stl[:, 1].min()) * 1000
ax3.text(15, -73, f'Y-depth: {y_span:.1f} mm\n(holes at different depths)',
         fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

ax3.set_xlabel('X (mm)', fontweight='bold')
ax3.set_ylabel('Y (mm)', fontweight='bold')
ax3.grid(True, alpha=0.3)

# View 4: Front view (YZ plane)
ax4 = fig.add_subplot(2, 3, 4)
ax4.set_title('Front View (YZ plane)', fontsize=12, fontweight='bold')

for i, (hole, label, color) in enumerate(zip(screw_holes_stl, labels, colors)):
    ax4.scatter(hole[1]*1000, hole[2]*1000, c=color, s=400, marker='o',
                edgecolors='black', linewidths=2, zorder=10)

ax4.scatter(center[1]*1000, center[2]*1000, c='purple', s=600, marker='*',
            edgecolors='black', linewidths=2, zorder=15)

ax4.set_xlabel('Y (mm)', fontweight='bold')
ax4.set_ylabel('Z (mm)', fontweight='bold')
ax4.grid(True, alpha=0.3)

# View 5: Screw hole pattern schematic
ax5 = fig.add_subplot(2, 3, 5)
ax5.set_title('Screw Hole Pattern (Schematic)', fontsize=12, fontweight='bold')

# Draw as clean square pattern
for i, (hole, label, color) in enumerate(zip(screw_holes_stl, labels, colors)):
    x, z = hole[0]*1000, hole[2]*1000

    # Draw screw hole as circle
    circle = plt.Circle((x, z), 1.5, color=color, ec='black', linewidth=2, zorder=10)
    ax5.add_patch(circle)

    # Add label outside
    offset_x = 6 if x > 0 else -6
    offset_z = 6 if z > 0 else -6
    ax5.text(x + offset_x, z + offset_z, label.replace('\n', '\n'),
             fontsize=11, ha='center', va='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.5, edgecolor='black', linewidth=1.5))

# Camera center
ax5.scatter(center[0]*1000, center[2]*1000, c='purple', s=800, marker='*',
            edgecolors='black', linewidths=3, zorder=15)
ax5.text(center[0]*1000, center[2]*1000 - 8, 'Camera\nLens Hole',
         fontsize=11, ha='center', va='top', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='purple', alpha=0.5, edgecolor='black', linewidth=1.5))

# Draw mounting surface outline
outline_x = [screw_holes_stl[0,0]*1000, screw_holes_stl[1,0]*1000,
             screw_holes_stl[3,0]*1000, screw_holes_stl[2,0]*1000, screw_holes_stl[0,0]*1000]
outline_z = [screw_holes_stl[0,2]*1000, screw_holes_stl[1,2]*1000,
             screw_holes_stl[3,2]*1000, screw_holes_stl[2,2]*1000, screw_holes_stl[0,2]*1000]
ax5.plot(outline_x, outline_z, 'k-', linewidth=4, alpha=0.5, zorder=1)

# Fill mounting surface
ax5.fill(outline_x, outline_z, color='lightgray', alpha=0.3, zorder=0)

ax5.set_xlabel('X (mm)', fontweight='bold')
ax5.set_ylabel('Z (mm)', fontweight='bold')
ax5.set_aspect('equal')
ax5.grid(True, alpha=0.3, linestyle='--')
ax5.set_xlim(-20, 25)
ax5.set_ylim(-16, 8)

# View 6: Summary info
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')

info_text = f"""
SCREW HOLE ANALYSIS SUMMARY

4 Corner Screw Holes (STL frame):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  H1 (bottom-left):  [{screw_holes_stl[0,0]:7.4f}, {screw_holes_stl[0,1]:7.4f}, {screw_holes_stl[0,2]:7.4f}] m
  H4 (top-left):     [{screw_holes_stl[1,0]:7.4f}, {screw_holes_stl[1,1]:7.4f}, {screw_holes_stl[1,2]:7.4f}] m
  H6 (bottom-right): [{screw_holes_stl[2,0]:7.4f}, {screw_holes_stl[2,1]:7.4f}, {screw_holes_stl[2,2]:7.4f}] m
  H9 (top-right):    [{screw_holes_stl[3,0]:7.4f}, {screw_holes_stl[3,1]:7.4f}, {screw_holes_stl[3,2]:7.4f}] m

Screw Hole Spacing:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  X (left-right): {x_span:.2f} mm
  Z (up-down):    {z_span:.2f} mm
  Y (depth):      {y_span:.2f} mm

Camera Position (at center):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STL frame:     [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}] m
  Gripper frame: [{center_gripper[0]:.4f}, {center_gripper[1]:.4f}, {center_gripper[2]:.4f}] m

XML Configuration:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<camera name="wrist_camera"
        pos="{center_gripper[0]:.4f} {center_gripper[1]:.4f} {center_gripper[2]:.4f}"
        axisangle="1.000000 0.000000 0.000000 1.134477"
        fovy="140"/>

Notes:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Camera positioned at geometric center of 4 screw holes
• Oriented perpendicular to mounting surface (65° pitch)
• Mounting surface ~32mm × 13mm (rectangular, not square)
• Y-depth variation shows holes at different depths in mount
"""

ax6.text(0.05, 0.5, info_text, fontsize=10, family='monospace',
         verticalalignment='center', transform=ax6.transAxes,
         bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8, edgecolor='black', linewidth=2))

plt.suptitle('Camera Mount: 4 Corner Screw Holes Analysis',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.97])

output_file = "four_screw_holes_visualization.png"
plt.savefig(output_file, dpi=200, bbox_inches='tight')
print(f"\n✅ Saved: {output_file}")
print("="*80)
