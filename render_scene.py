#!/usr/bin/env python
"""
Simple script to render the SO101 scene and save an image to verify paper visibility.
"""

import sys
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load the XML file
xml_path = Path("src/lerobot/envs/so101_assets/paper_square_realistic.xml")
print(f"Loading MuJoCo model from {xml_path}...")
model = mj.MjModel.from_xml_path(str(xml_path))
data = mj.MjData(model)

# Set robot to home position
joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
home_pos = np.array([0.0, 0.3, -0.6, 0.0, 0.0, 0.0])

for i, joint_name in enumerate(joint_names):
    joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    data.qpos[model.jnt_qposadr[joint_id]] = home_pos[i]

# Get paper position
paper_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "paper")
paper_joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "paper_free")
paper_qpos_addr = model.jnt_qposadr[paper_joint_id]

print(f"\nPaper initial position in XML:")
print(f"  Position: {data.qpos[paper_qpos_addr:paper_qpos_addr+3]}")
print(f"  Quaternion: {data.qpos[paper_qpos_addr+3:paper_qpos_addr+7]}")

# Run forward dynamics
mj.mj_forward(model, data)

print(f"\nPaper position after mj_forward:")
print(f"  Body position: {data.xpos[paper_body_id]}")

# Create renderer
renderer = mj.Renderer(model, height=480, width=640)

# Update scene
renderer.update_scene(data, camera="top_view")

# Render image
pixels = renderer.render()

# Save image
img = Image.fromarray(pixels)
output_path = "scene_render.png"
img.save(output_path)
print(f"\nRendered image saved to: {output_path}")
print("Check this image to see if the white paper is visible!")

# Print paper geom info
paper_geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "paper_geom")
print(f"\nPaper geom info:")
print(f"  ID: {paper_geom_id}")
print(f"  Size: {model.geom_size[paper_geom_id]}")
print(f"  RGBA: {model.geom_rgba[paper_geom_id]}")
print(f"  Type: {model.geom_type[paper_geom_id]}")
