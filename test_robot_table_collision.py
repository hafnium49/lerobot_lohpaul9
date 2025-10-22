import sys
sys.path.insert(0, 'src')
import mujoco as mj
import numpy as np

# Load model
model = mj.MjModel.from_xml_path('src/lerobot/envs/so101_assets/paper_square_realistic.xml')
data = mj.MjData(model)

# Set robot to home position
joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
home_pos = np.array([0.0, 0.3, -0.6, 0.0, 0.0, 0.0])

for i, joint_name in enumerate(joint_names):
    joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    data.qpos[model.jnt_qposadr[joint_id]] = home_pos[i]

# Forward kinematics
mj.mj_forward(model, data)

print("Testing robot-table collision...")
print(f"Number of contacts before pushing robot down: {data.ncon}")

# Now apply a big downward action on elbow to try to push it through the table
elbow_joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "elbow_flex")
shoulder_joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "shoulder_lift")

# Push elbow way down
data.qpos[model.jnt_qposadr[elbow_joint_id]] = -2.5  # Very low elbow
data.qpos[model.jnt_qposadr[shoulder_joint_id]] = 0.8  # Shoulder up

# Run physics
for _ in range(100):
    mj.mj_step(model, data)

print(f"Number of contacts after pushing robot down: {data.ncon}")

# Check contact details
if data.ncon > 0:
    print("\nContacts detected:")
    for i in range(min(data.ncon, 10)):
        contact = data.contact[i]
        geom1_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom_{contact.geom1}"
        geom2_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom_{contact.geom2}"
        print(f"  {geom1_name} <-> {geom2_name} (dist: {contact.dist:.4f}m)")
else:
    print("\n❌ NO CONTACTS - Robot is passing through table!")

# Check elbow position
elbow_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "elbow")
elbow_z = data.xpos[elbow_body_id][2]
print(f"\nElbow Z position: {elbow_z:.4f}m")
if elbow_z < 0:
    print("❌ Elbow is BELOW the table (Z < 0)!")
else:
    print("✅ Elbow stayed above table")
