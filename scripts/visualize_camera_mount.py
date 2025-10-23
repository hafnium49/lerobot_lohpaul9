#!/usr/bin/env python3
"""Visualize the camera mount and camera views."""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.envs.so101_residual_env import SO101ResidualEnv


def main():
    print("Creating environment with camera mount...")
    env = SO101ResidualEnv(render_mode="rgb_array")
    obs, info = env.reset()
    print("✅ Environment created successfully!")

    # Get model and data
    model = env.model
    data = env.data

    # Create renderers for different views
    print("\nRendering camera views...")

    # Get camera IDs
    top_view_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "top_view")
    wrist_cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "wrist_camera")

    # Render from both cameras
    renderer = mj.Renderer(model, 480, 640)

    # Top view
    renderer.update_scene(data, camera=top_view_id)
    top_view_img = renderer.render()

    # Wrist camera view
    renderer.update_scene(data, camera=wrist_cam_id)
    wrist_view_img = renderer.render()

    # Side view - just render from default free camera
    renderer_side = mj.Renderer(model, 480, 640)
    renderer_side.update_scene(data, camera=-1)  # Free camera default view
    side_view_img = renderer_side.render()

    print("✅ Rendered all views!")

    # Create composite image
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top view
    axes[0, 0].imshow(top_view_img)
    axes[0, 0].set_title('Top View Camera\n(Position: [0.275, 0.175, 0.400] m)', fontsize=10)
    axes[0, 0].axis('off')

    # Wrist camera view
    axes[0, 1].imshow(wrist_view_img)
    axes[0, 1].set_title('Wrist Camera View\n(Mounted on camera mount)', fontsize=10)
    axes[0, 1].axis('off')

    # Side view showing camera mount
    axes[1, 0].imshow(side_view_img)
    axes[1, 0].set_title('Side View\n(Showing camera mount on gripper)', fontsize=10)
    axes[1, 0].axis('off')

    # Camera info text
    axes[1, 1].axis('off')
    camera_info = f"""
CAMERA MOUNT INTEGRATION

✅ Fixed jaw replaced with:
   Wrist_Cam_Mount_32x32_UVC_Module_SO101_MIRRORED.stl

📷 Top View Camera:
   Position: [0.275, 0.175, 0.400] m
   Direction: Looking straight down
   FOV: 140°

📷 Wrist Camera:
   Position: [0.314, 0.143, 0.251] m
   Direction: [0.604, 0.637, -0.479]
   FOV: 140°
   Mounted on: Camera mount extension arm

🎯 Camera Mount Features:
   - Replaces original fixed jaw
   - Integrated camera mounting bracket
   - Mirrored for correct side placement
   - Maintains gripper functionality
    """
    axes[1, 1].text(0.1, 0.5, camera_info,
                    fontsize=9, family='monospace',
                    verticalalignment='center')

    plt.tight_layout()

    output_file = "camera_mount_visualization.png"
    print(f"\nSaving visualization to: {output_file}")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")

    env.close()
    print("\n✅ Complete!")


if __name__ == "__main__":
    main()
