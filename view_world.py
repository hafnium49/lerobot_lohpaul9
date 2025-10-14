#!/usr/bin/env python
"""
Launch MuJoCo interactive viewer for SO-101 paper-square world.

Usage:
    python view_world.py

Controls:
    - Mouse drag: Rotate view
    - Scroll: Zoom in/out
    - Right-click drag: Pan view
    - Double-click: Select body
    - Ctrl+Right-click: Apply force to body
    - Space: Pause/resume simulation
    - Tab: Toggle left panel
    - Shift+Tab: Toggle right panel
    - ESC or close window: Exit
"""

import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path

# Path to your world
xml_path = Path("src/lerobot/envs/so101_assets/paper_square.xml")

print("=" * 60)
print("SO-101 Paper-Square MuJoCo Viewer")
print("=" * 60)
print(f"\nLoading model: {xml_path}")
print(f"File exists: {xml_path.exists()}\n")

if not xml_path.exists():
    print(f"❌ Error: Model file not found at {xml_path}")
    print("   Make sure you're running from the repository root.")
    exit(1)

# Load model
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

print("✅ Model loaded successfully!")
print(f"\nWorld info:")
print(f"  • Bodies: {model.nbody}")
print(f"  • Joints: {model.njnt}")
print(f"  • DOFs: {model.nv}")
print(f"  • Actuators: {model.nu}")
print(f"  • Physics timestep: {model.opt.timestep:.6f} s ({1/model.opt.timestep:.0f} Hz)")

# Set robot to a nice initial pose
mujoco.mj_resetData(model, data)

# Home position for robot
joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper_left_joint"]
home_pos = [0.0, 0.3, -0.6, -np.pi/2, 0.0, 0.005]

for name, pos in zip(joint_names, home_pos):
    try:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0:
            data.qpos[joint_id] = pos
    except:
        pass

# Position paper in front of robot
try:
    paper_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "paper_free")
    if paper_joint_id >= 0:
        paper_qpos_addr = model.jnt_qposadr[paper_joint_id]
        # Position: between robot and target
        data.qpos[paper_qpos_addr:paper_qpos_addr+3] = [0.30, 0.0, 0.001]
        # Orientation: identity quaternion
        data.qpos[paper_qpos_addr+3:paper_qpos_addr+7] = [1, 0, 0, 0]
except:
    pass

# Forward kinematics
mujoco.mj_forward(model, data)

# Get initial positions for info
try:
    paper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paper")
    tape_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tape_square")
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    paper_pos = data.xpos[paper_body_id]
    tape_pos = data.xpos[tape_body_id]
    ee_pos = data.site_xpos[ee_site_id]

    dist = np.linalg.norm(paper_pos[:2] - tape_pos[:2])

    print(f"\nInitial configuration:")
    print(f"  • Paper: ({paper_pos[0]:.3f}, {paper_pos[1]:.3f}, {paper_pos[2]:.3f}) m")
    print(f"  • Target: ({tape_pos[0]:.3f}, {tape_pos[1]:.3f}) m")
    print(f"  • End-effector: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}) m")
    print(f"  • Distance to goal: {dist*100:.1f} cm")
except:
    pass

print("\n" + "=" * 60)
print("Viewer Controls:")
print("=" * 60)
print("  Mouse drag           - Rotate camera")
print("  Scroll               - Zoom in/out")
print("  Right-click drag     - Pan camera")
print("  Double-click         - Select body (shows info)")
print("  Ctrl+Right-click     - Apply force to body")
print("  Space                - Pause/resume simulation")
print("  Tab                  - Toggle left panel")
print("  Shift+Tab            - Toggle right panel")
print("  [ and ]              - Decrease/increase simulation speed")
print("  Backspace            - Reset simulation")
print("  ESC or close window  - Exit")
print("=" * 60)

print("\n🚀 Launching interactive viewer...")
print("   (Close the viewer window or press ESC to exit)\n")

# Launch the viewer (blocking call)
try:
    mujoco.viewer.launch(model, data)
    print("\n✅ Viewer closed normally.")

except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user (Ctrl+C)")

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("  • On headless systems, set: export MUJOCO_GL=egl")
    print("  • On macOS, you may need to use: mjpython view_world.py")
    print("  • Check that your display is configured correctly")
