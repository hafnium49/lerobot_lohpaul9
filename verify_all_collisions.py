#!/usr/bin/env python
"""
Verify collision settings for ALL geoms in the SO-101 model.
"""

import sys
sys.path.insert(0, 'src')
import mujoco as mj

# Load model
model = mj.MjModel.from_xml_path('src/lerobot/envs/so101_assets/paper_square_realistic.xml')
data = mj.MjData(model)

print("="*80)
print("COMPLETE GEOM COLLISION ANALYSIS")
print("="*80)

# Group geoms by body
body_geoms = {}
for geom_id in range(model.ngeom):
    body_id = model.geom_bodyid[geom_id]
    if body_id not in body_geoms:
        body_geoms[body_id] = []
    body_geoms[body_id].append(geom_id)

# Analyze each body
for body_id in sorted(body_geoms.keys()):
    body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"

    # Skip non-robot bodies
    if body_name in ["world", "table", "paper", "tape_square", "markers"]:
        continue

    print(f"\n{body_name.upper()}:")

    has_collision = False
    for geom_id in body_geoms[body_id]:
        geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
        contype = model.geom_contype[geom_id]
        conaffinity = model.geom_conaffinity[geom_id]
        group = model.geom_group[geom_id]

        # Check if it collides with table (table has contype=2, conaffinity=1)
        table_collision = bool((contype & 1) or (2 & conaffinity))

        if contype > 0:
            has_collision = True
            status = "✅ COLLIDES" if table_collision else "⚠️  NO TABLE COLLISION"
            print(f"  {geom_name:30s} group={group} contype={contype} conaffinity={conaffinity} {status}")

    if not has_collision:
        print(f"  ❌ NO COLLISION GEOMS - WILL PASS THROUGH TABLE!")

print("\n" + "="*80)
print("TABLE COLLISION SETTINGS:")
print("="*80)
# Find table geom
for geom_id in range(model.ngeom):
    geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id) or ""
    if "table" in geom_name.lower():
        print(f"{geom_name}: contype={model.geom_contype[geom_id]} conaffinity={model.geom_conaffinity[geom_id]}")

print("\n" + "="*80)
print("COLLISION FORMULA:")
print("="*80)
print("Robot-table collision happens when:")
print("  (robot.contype & table.conaffinity) OR (table.contype & robot.conaffinity)")
print("  (robot.contype & 1) OR (2 & robot.conaffinity)")
print("="*80)
