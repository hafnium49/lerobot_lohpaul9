#!/usr/bin/env python3
"""
Test the optimized friction settings for paper manipulation.

Verifies:
1. Paper slides easily on table (low friction)
2. Fingertips have high grip on paper (high friction)
3. Friction hierarchy is correct
"""

import mujoco
import numpy as np
from pathlib import Path

# Load the model
model_path = Path("src/lerobot/envs/so101_assets/paper_square_realistic.xml")
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

print("=" * 80)
print("FRICTION OPTIMIZATION VERIFICATION")
print("=" * 80)

# Get geom IDs
paper_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "paper_geom")
table_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_surface")
fixed_fingertip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_fingertip")
moving_fingertip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_fingertip")

print("\n📊 FRICTION COEFFICIENTS (μ_slide):")
print("-" * 80)

# Paper friction
paper_friction = model.geom_friction[paper_geom_id]
print(f"Paper:            μ = {paper_friction[0]:.2f}  (slide={paper_friction[0]:.3f}, spin={paper_friction[1]:.4f}, roll={paper_friction[2]:.6f})")

# Table friction
table_friction = model.geom_friction[table_geom_id]
print(f"Table:            μ = {table_friction[0]:.2f}  (slide={table_friction[0]:.3f}, spin={table_friction[1]:.4f}, roll={table_friction[2]:.6f})")

# Fingertip friction
fixed_friction = model.geom_friction[fixed_fingertip_id]
moving_friction = model.geom_friction[moving_fingertip_id]
print(f"Fixed Fingertip:  μ = {fixed_friction[0]:.2f}  (slide={fixed_friction[0]:.3f}, spin={fixed_friction[1]:.4f}, roll={fixed_friction[2]:.6f})")
print(f"Moving Fingertip: μ = {moving_friction[0]:.2f}  (slide={moving_friction[0]:.3f}, spin={moving_friction[1]:.4f}, roll={moving_friction[2]:.6f})")

print("\n🎯 FRICTION HIERARCHY (effective contact friction):")
print("-" * 80)

# Effective friction (approximate as min)
paper_table_eff = min(paper_friction[0], table_friction[0])
fingertip_paper_eff = min(fixed_friction[0], paper_friction[0])

print(f"Paper-Table contact:    μ_eff ≈ min({paper_friction[0]:.2f}, {table_friction[0]:.2f}) = {paper_table_eff:.2f}")
print(f"Fingertip-Paper contact: μ_eff ≈ min({fixed_friction[0]:.2f}, {paper_friction[0]:.2f}) = {fingertip_paper_eff:.2f}")

# Friction ratio
friction_ratio = fingertip_paper_eff / paper_table_eff if paper_table_eff > 0 else float('inf')
print(f"\n📈 Fingertip/Table friction ratio: {friction_ratio:.1f}×")

print("\n✅ VERIFICATION:")
print("-" * 80)

# Check paper friction is low
if paper_friction[0] <= 0.20:
    print(f"✅ Paper friction LOW (μ={paper_friction[0]:.2f} ≤ 0.20) - slides easily on table")
else:
    print(f"❌ Paper friction TOO HIGH (μ={paper_friction[0]:.2f} > 0.20) - will stick to table")

# Check table friction is low
if table_friction[0] <= 0.25:
    print(f"✅ Table friction LOW (μ={table_friction[0]:.2f} ≤ 0.25) - allows smooth paper motion")
else:
    print(f"❌ Table friction TOO HIGH (μ={table_friction[0]:.2f} > 0.25) - will resist paper sliding")

# Check fingertip friction is high
if fixed_friction[0] >= 1.5:
    print(f"✅ Fingertip friction HIGH (μ={fixed_friction[0]:.2f} ≥ 1.5) - firm grip on paper")
else:
    print(f"❌ Fingertip friction TOO LOW (μ={fixed_friction[0]:.2f} < 1.5) - weak grip")

# Check friction hierarchy
if fingertip_paper_eff > paper_table_eff * 1.5:
    print(f"✅ Friction hierarchy CORRECT ({friction_ratio:.1f}× advantage for grip)")
else:
    print(f"❌ Friction hierarchy INCORRECT (ratio {friction_ratio:.1f}× too low)")

print("\n💡 EXPECTED BEHAVIOR:")
print("-" * 80)
print("• Paper should slide easily on table with small robot force")
print("• Fingertips should NOT slip when pushing/gripping paper")
print("• Robot can push paper across table smoothly")
print("• Grasping creates firm contact (no slipping during lift)")

print("\n🔬 PHYSICAL INTERPRETATION:")
print("-" * 80)
print(f"Paper-Table:    {paper_table_eff:.2f} → {'Very low friction (like paper on smooth plastic)' if paper_table_eff < 0.2 else 'Moderate friction'}")
print(f"Fingertip-Paper: {fingertip_paper_eff:.2f} → {'Low-medium (allows controlled sliding)' if fingertip_paper_eff < 0.3 else 'High friction (firm grip)'}")
print(f"Fingertip (raw):  {fixed_friction[0]:.2f} → {'Rubber-like grip (realistic for soft fingertips)' if fixed_friction[0] > 1.5 else 'Too slippery'}")

print("\n" + "=" * 80)
print("Test complete!")
print("=" * 80)
