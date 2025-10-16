#!/usr/bin/env python
"""
Visualize camera views from SO-101 paper-square world.

This script renders offscreen RGB images from both cameras:
- top_view: Bird's eye view from 400mm above target
- wrist_camera: Egocentric view from gripper

Shows what GR00T N1.5 will actually see as visual input.
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Path to world XML
xml_path = Path("src/lerobot/envs/so101_assets/paper_square_realistic.xml")

print("=" * 60)
print("SO-101 Camera View Visualizer")
print("=" * 60)
print(f"\nLoading model: {xml_path}")

if not xml_path.exists():
    print(f"❌ Error: Model file not found at {xml_path}")
    exit(1)

# Load model
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

print("✅ Model loaded successfully!")

# Check cameras
print(f"\n📷 Cameras in model: {model.ncam}")
for i in range(model.ncam):
    cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
    print(f"  {i}: {cam_name}")

# Set robot to home position
mujoco.mj_resetData(model, data)

joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
home_pos = [0.0, 0.3, -0.6, -np.pi/2, 0.0, 0.005]

for name, pos in zip(joint_names, home_pos):
    try:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0:
            data.qpos[joint_id] = pos
    except:
        pass

# Position paper
try:
    paper_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "paper_free")
    if paper_joint_id >= 0:
        paper_qpos_addr = model.jnt_qposadr[paper_joint_id]
        data.qpos[paper_qpos_addr:paper_qpos_addr+3] = [0.025, 0.175, 0.001]
        data.qpos[paper_qpos_addr+3:paper_qpos_addr+7] = [1, 0, 0, 0]
except:
    pass

# Forward kinematics
mujoco.mj_forward(model, data)

print("\n🎨 Rendering camera views...")

# Image dimensions
width, height = 640, 480

# Create renderer
renderer = mujoco.Renderer(model, height=height, width=width)

# Function to render from a specific camera
def render_camera(camera_name):
    """Render RGB image from specified camera."""
    try:
        # Get camera ID
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

        # Update scene
        renderer.update_scene(data, camera=cam_id)

        # Render RGB image
        rgb = renderer.render()

        return rgb
    except Exception as e:
        print(f"❌ Error rendering {camera_name}: {e}")
        return None

# Render both cameras
top_view_img = render_camera("top_view")
wrist_img = render_camera("wrist_camera")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor('black')

if top_view_img is not None:
    axes[0].imshow(top_view_img)
    axes[0].set_title('Top View Camera\n(400mm above target, 140° FOV)',
                      color='white', fontsize=14, weight='bold', pad=20)
    axes[0].axis('off')

    # Save image
    plt.imsave('top_view_camera.png', top_view_img)
    print("✅ Saved: top_view_camera.png")
else:
    axes[0].text(0.5, 0.5, 'Top View\nNot Available',
                ha='center', va='center', color='red', fontsize=16)
    axes[0].set_facecolor('black')
    axes[0].axis('off')

if wrist_img is not None:
    axes[1].imshow(wrist_img)
    axes[1].set_title('Wrist Camera\n(Egocentric view, 140° FOV)',
                      color='white', fontsize=14, weight='bold', pad=20)
    axes[1].axis('off')

    # Save image
    plt.imsave('wrist_camera.png', wrist_img)
    print("✅ Saved: wrist_camera.png")
else:
    axes[1].text(0.5, 0.5, 'Wrist Camera\nNot Available',
                ha='center', va='center', color='red', fontsize=16)
    axes[1].set_facecolor('black')
    axes[1].axis('off')

# Overall title
fig.suptitle('GR00T N1.5 Vision Input: Camera Views',
             color='white', fontsize=16, weight='bold', y=0.98)

plt.tight_layout()
plt.savefig('camera_views.png', dpi=150, facecolor='black', edgecolor='none')
print("✅ Saved: camera_views.png (side-by-side comparison)")

print("\n" + "=" * 60)
print("Camera Views Rendered Successfully!")
print("=" * 60)
print("\nFiles created:")
print("  • top_view_camera.png - Bird's eye view")
print("  • wrist_camera.png - Egocentric view")
print("  • camera_views.png - Side-by-side comparison")
print("\n💡 These are the RGB images GR00T N1.5 will receive as visual input.")
print("\n🖼️  Displaying visualization...")

plt.show()

print("\n✅ Done!")
