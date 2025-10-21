#!/usr/bin/env python
"""
Launch MuJoCo interactive viewer for SO-101 with REAL-TIME CONTACT DETECTION.

Usage:
    python view_world_with_contacts.py

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

Real-Time Feedback:
    - Terminal shows contact events as they happen
    - 🟢 GREEN: Fingertip-paper contact (GOOD - +0.1 reward)
    - 🟡 YELLOW: Robot-paper contact (WARNING - -0.2 penalty)
    - 🔴 RED: Robot-table contact (BAD - -0.5 penalty)
"""

import time
from pathlib import Path

import mujoco
import numpy as np

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# Path to world
xml_path = Path("src/lerobot/envs/so101_assets/paper_square_realistic.xml")

print("=" * 80)
print(f"{Colors.BOLD}SO-101 Paper-Square MuJoCo Viewer with Contact Detection{Colors.RESET}")
print("=" * 80)
print(f"\nLoading model: {xml_path}")

if not xml_path.exists():
    print(f"❌ Error: Model file not found at {xml_path}")
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

# ============================================================================
# Setup Contact Detection (same logic as so101_residual_env.py)
# ============================================================================

print(f"\n{Colors.CYAN}Setting up contact detection...{Colors.RESET}")

# Get body IDs
paper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paper")
tape_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tape_square")
ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

# Get fingertip geom IDs
fixed_fingertip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_fingertip")
moving_fingertip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_fingertip")

# Get table geom ID
table_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_surface")

# Get all robot collision geoms (for table/paper contact penalty)
robot_body_names = ["shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"]
robot_collision_geom_ids = []

for geom_id in range(model.ngeom):
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""

    # Only consider geoms with collision enabled (contype > 0)
    # Exclude fingertips from penalty
    if (model.geom_contype[geom_id] > 0 and
        model.geom_group[geom_id] == 0 and
        "fingertip" not in geom_name):

        body_id = model.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""

        # Only robot bodies
        if body_name in robot_body_names:
            robot_collision_geom_ids.append(geom_id)

print(f"  ✅ Fingertip geoms: {fixed_fingertip_id}, {moving_fingertip_id}")
print(f"  ✅ Table geom: {table_geom_id}")
print(f"  ✅ Robot collision geoms: {len(robot_collision_geom_ids)} geoms tracked")
print(f"  ✅ Paper body: {paper_body_id}")

# Check fingertip sizes
if fixed_fingertip_id >= 0:
    ft_size = model.geom_size[fixed_fingertip_id][0]
    print(f"  ✅ Fingertip diameter: {ft_size*2*1000:.1f}mm")

def detect_contacts(data, model):
    """
    Detect contacts between fingertips/robot and paper/table.
    Returns dict with contact types and details.
    """
    robot_table = []
    robot_paper = []
    fingertip_paper = []

    # Iterate through all active contacts
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2

        # Get geom names for display
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or f"geom_{geom1}"
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or f"geom_{geom2}"

        # Robot-table contact
        if ((geom1 in robot_collision_geom_ids and geom2 == table_geom_id) or
            (geom2 in robot_collision_geom_ids and geom1 == table_geom_id)):
            robot_table.append((name1, name2))

        # Paper geoms
        paper_geom1 = model.geom_bodyid[geom1] == paper_body_id
        paper_geom2 = model.geom_bodyid[geom2] == paper_body_id

        # Robot-paper contact (excluding fingertips)
        if ((geom1 in robot_collision_geom_ids and paper_geom2) or
            (geom2 in robot_collision_geom_ids and paper_geom1)):
            robot_paper.append((name1, name2))

        # Fingertip-paper contact
        if fixed_fingertip_id >= 0 and moving_fingertip_id >= 0:
            if ((geom1 in [fixed_fingertip_id, moving_fingertip_id] and paper_geom2) or
                (geom2 in [fixed_fingertip_id, moving_fingertip_id] and paper_geom1)):
                fingertip_paper.append((name1, name2))

    return {
        "robot_table": robot_table,
        "robot_paper": robot_paper,
        "fingertip_paper": fingertip_paper,
    }

# ============================================================================
# Initialize Robot Position
# ============================================================================

mujoco.mj_resetData(model, data)

# Home position
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

# Print initial configuration
paper_pos = data.xpos[paper_body_id]
tape_pos = data.xpos[tape_body_id]
ee_pos = data.site_xpos[ee_site_id]
dist = np.linalg.norm(paper_pos[:2] - tape_pos[:2])

print(f"\n{Colors.CYAN}Initial configuration:{Colors.RESET}")
print(f"  • Paper: ({paper_pos[0]:.3f}, {paper_pos[1]:.3f}, {paper_pos[2]:.3f}) m")
print(f"  • Target: ({tape_pos[0]:.3f}, {tape_pos[1]:.3f}) m")
print(f"  • End-effector: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}) m")
print(f"  • Distance to goal: {dist*100:.1f} cm")

# ============================================================================
# Viewer Controls Info
# ============================================================================

print("\n" + "=" * 80)
print(f"{Colors.BOLD}Viewer Controls:{Colors.RESET}")
print("=" * 80)
print("  Mouse drag           - Rotate camera")
print("  Scroll               - Zoom in/out")
print("  Right-click drag     - Pan camera")
print("  Double-click         - Select body (shows info)")
print(f"  {Colors.BOLD}Ctrl+Right-click     - Apply force to body (USE THIS TO MOVE ROBOT!){Colors.RESET}")
print("  Space                - Pause/resume simulation")
print("  Tab                  - Toggle left panel")
print("  Shift+Tab            - Toggle right panel")
print("  Backspace            - Reset simulation")
print("  ESC or close window  - Exit")
print("=" * 80)

print("\n" + "=" * 80)
print(f"{Colors.BOLD}Contact Detection Legend:{Colors.RESET}")
print("=" * 80)
print(f"  {Colors.GREEN}🟢 GREEN{Colors.RESET}  - Fingertip-paper contact (GOOD! +0.1 reward)")
print(f"  {Colors.YELLOW}🟡 YELLOW{Colors.RESET} - Robot-paper contact (WARNING - -0.2 penalty)")
print(f"  {Colors.RED}🔴 RED{Colors.RESET}    - Robot-table contact (BAD - -0.5 penalty)")
print("=" * 80)

print(f"\n{Colors.CYAN}🚀 Launching interactive viewer with real-time contact detection...{Colors.RESET}")
print(f"{Colors.BOLD}TIP: Use Ctrl+Right-click and drag to move robot parts toward the paper!{Colors.RESET}\n")

# ============================================================================
# Custom Viewer Loop with Contact Detection
# ============================================================================

# Track previous contact state to only print on changes
prev_contacts = {
    "robot_table": [],
    "robot_paper": [],
    "fingertip_paper": [],
}

# Launch viewer in passive mode
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Simulation loop
    step_count = 0

    while viewer.is_running():
        # Step physics
        mujoco.mj_step(model, data)

        # Detect contacts every 10 steps (reduce terminal spam)
        if step_count % 10 == 0:
            contacts = detect_contacts(data, model)

            # Check for new contact events
            # Fingertip-paper (GOOD)
            if contacts["fingertip_paper"] and not prev_contacts["fingertip_paper"]:
                print(f"\n{Colors.GREEN}{'='*80}")
                print(f"🟢 FINGERTIP-PAPER CONTACT DETECTED! (Reward: +0.1)")
                print(f"{'='*80}{Colors.RESET}")
                for name1, name2 in contacts["fingertip_paper"]:
                    print(f"   {name1} <-> {name2}")
                print(f"{Colors.GREEN}{'='*80}{Colors.RESET}\n")

            # Robot-paper (WARNING)
            if contacts["robot_paper"] and not prev_contacts["robot_paper"]:
                print(f"\n{Colors.YELLOW}{'='*80}")
                print(f"🟡 ROBOT-PAPER CONTACT (Penalty: -0.2)")
                print(f"{'='*80}{Colors.RESET}")
                for name1, name2 in contacts["robot_paper"]:
                    print(f"   {name1} <-> {name2}")
                print(f"{Colors.YELLOW}{'='*80}{Colors.RESET}\n")

            # Robot-table (BAD)
            if contacts["robot_table"] and not prev_contacts["robot_table"]:
                print(f"\n{Colors.RED}{'='*80}")
                print(f"🔴 ROBOT-TABLE CONTACT! (Penalty: -0.5)")
                print(f"{'='*80}{Colors.RESET}")
                for name1, name2 in contacts["robot_table"]:
                    print(f"   {name1} <-> {name2}")
                print(f"{Colors.RED}{'='*80}{Colors.RESET}\n")

            # Contact ended messages
            if prev_contacts["fingertip_paper"] and not contacts["fingertip_paper"]:
                print(f"{Colors.GREEN}✓ Fingertip-paper contact ended{Colors.RESET}")

            if prev_contacts["robot_paper"] and not contacts["robot_paper"]:
                print(f"{Colors.YELLOW}✓ Robot-paper contact ended{Colors.RESET}")

            if prev_contacts["robot_table"] and not contacts["robot_table"]:
                print(f"{Colors.RED}✓ Robot-table contact ended{Colors.RESET}")

            # Update previous state
            prev_contacts = contacts

        # Sync viewer
        viewer.sync()

        step_count += 1

        # Small sleep to prevent busy loop
        time.sleep(0.001)

print(f"\n{Colors.CYAN}✅ Viewer closed normally.{Colors.RESET}")
