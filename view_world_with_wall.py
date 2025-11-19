#!/usr/bin/env python3
"""
Render the SO-101 MuJoCo world with the new white wall and save an image.
"""

import mujoco as mj
import mujoco.viewer
import numpy as np
import cv2
import os

# Load the model with the new wall
xml_path = "src/lerobot/envs/so101_assets/paper_square_realistic.xml"
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# Set robot to home position
qpos_home = [0.0, 0.0, -0.866, -0.87, 0.0, 0.0]  # Home position
data.qpos[:6] = qpos_home

# Position paper in front of robot
paper_start_pos = [0.26, 0.10, 0.001]  # Paper position
paper_start_quat = [1, 0, 0, 0]  # Upright orientation

# Set paper pose
paper_joint_id = model.joint("paper_free").id
paper_qpos_addr = model.jnt_qposadr[paper_joint_id]
data.qpos[paper_qpos_addr:paper_qpos_addr+3] = paper_start_pos
data.qpos[paper_qpos_addr+3:paper_qpos_addr+7] = paper_start_quat

# Forward dynamics
mj.mj_forward(model, data)

# Create renderer for top view (limited by framebuffer)
renderer = mj.Renderer(model, height=480, width=640)

# Render from top camera
cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "top_view")
if cam_id >= 0:
    renderer.update_scene(data, camera=cam_id)
    pixels = renderer.render()

    # Save the image
    output_path = "world_with_wall_topview.png"
    cv2.imwrite(output_path, cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    print(f"Saved top view to {output_path}")
else:
    print("Top view camera not found")

# Also render from a custom side angle to show the wall better
renderer2 = mj.Renderer(model, height=480, width=640)

# Update scene from default free camera
renderer2.update_scene(data)
# Set custom camera position to see the wall
renderer2.scene.lookat = [0.3, 0.175, 0.1]  # Look at center of workspace
renderer2.scene.distance = 0.8  # Distance from lookat point
renderer2.scene.elevation = -20  # Camera elevation angle
renderer2.scene.azimuth = 45  # Camera azimuth angle

pixels2 = renderer2.render()

# Save the angled view
output_path2 = "world_with_wall_angle.png"
cv2.imwrite(output_path2, cv2.cvtColor(pixels2, cv2.COLOR_RGB2BGR))
print(f"Saved angled view to {output_path2}")

print("\n=== World Configuration ===")
print(f"Red square center: [{0.275:.3f}, {0.175:.3f}] m")
print(f"Red square outer dimensions: 25.0 x 18.8 cm")
print(f"Right edge of red square: X = {0.390:.3f} m")
print(f"White wall position: X = {0.570:.3f} m (18 cm from right edge)")
print(f"Wall dimensions: 2cm thick x 60cm wide x 30cm tall")
print("\nThe white wall is positioned as a barrier 18cm beyond the right edge of the red target square.")
print("This prevents the paper from being pushed too far past the target area.")