#!/usr/bin/env python
"""List all bodies in the SO-101 model."""

import sys
sys.path.insert(0, 'src')
import mujoco as mj

model = mj.MjModel.from_xml_path('src/lerobot/envs/so101_assets/paper_square_realistic.xml')

print("="*80)
print("ALL BODIES IN MODEL")
print("="*80)

for body_id in range(model.nbody):
    body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
    print(f"{body_id:3d}: {body_name}")
