#!/usr/bin/env python
"""
Debug script to track robot body heights during policy execution.
"""

import sys
from pathlib import Path

import mujoco as mj
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.envs.so101_base_policy import JacobianIKPolicy
from lerobot.envs.so101_residual_env import SO101ResidualEnv

# Load model
model_path = "runs/baseline_pure_rl_fixed/so101_residual_zero_20251021_153009/final_model.zip"
print(f"Loading model from {model_path}...")
model = PPO.load(model_path)

# Create environment WITHOUT rendering
base_policy = JacobianIKPolicy(max_delta=0.02)
env = SO101ResidualEnv(
    base_policy=base_policy,
    alpha=1.0,
    act_scale=0.02,
    residual_penalty=0.0,
    randomize=True,
    render_mode=None,  # No rendering needed
    seed=42,
)

# Track robot body heights
obs, info = env.reset()

# Get all robot body IDs
body_names = [
    "base_link", "shoulder", "upper_arm", "elbow", "forearm",
    "wrist_link", "gripper_base", "gripper_left", "gripper_right"
]

print("\n" + "="*70)
print("Tracking robot body heights during episode")
print("="*70)

min_heights = {name: float('inf') for name in body_names}

for step in range(200):  # One episode
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # Check body heights every 10 steps
    if step % 10 == 0:
        print(f"\nStep {step}:")
        for body_name in body_names:
            try:
                body_id = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_BODY, body_name)
                z_pos = env.data.xpos[body_id][2]
                min_heights[body_name] = min(min_heights[body_name], z_pos)

                # Flag if below table
                flag = "❌ BELOW TABLE!" if z_pos < 0 else ""
                print(f"  {body_name:15s}: Z = {z_pos:+.4f}m  {flag}")
            except:
                pass

    if terminated or truncated:
        break

print("\n" + "="*70)
print("MINIMUM HEIGHTS REACHED:")
print("="*70)
for body_name, min_z in sorted(min_heights.items(), key=lambda x: x[1]):
    flag = "❌ PENETRATED TABLE!" if min_z < 0 else "✅"
    print(f"{body_name:15s}: {min_z:+.4f}m  {flag}")

env.close()
