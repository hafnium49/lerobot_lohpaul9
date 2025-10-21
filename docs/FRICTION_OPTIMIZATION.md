# Friction Optimization for Paper Manipulation

## Problem Statement

User observation: "The friction between the paper and table is too large and that between the paper and finger tip spheres is too small."

## Current Configuration (INCORRECT)

| Surface Pair | Friction (μ) | Issue |
|--------------|--------------|-------|
| Paper-Table | 0.60 × 0.30 = **effective high** | Paper sticks to table, hard to slide |
| Fingertip-Paper | 1.0 × 0.60 = **0.60** | Limited grip, fingers slip on paper |

### Current Settings:
```xml
<!-- Paper -->
<geom name="paper_geom" friction="0.60 0.001 0.00005"/>

<!-- Table -->
<geom name="table_surface" friction="0.30 0.001 0.00005"/>

<!-- Fingertips -->
<geom name="fixed_fingertip" friction="1.0 0.003 0.0001"/>
<geom name="moving_fingertip" friction="1.0 0.003 0.0001"/>
```

## Desired Behavior

For successful paper manipulation:

1. **Paper should slide easily on table** - Low friction (μ < 0.20)
2. **Fingertips should grip paper firmly** - High friction (μ > 1.2)
3. **Fingertips can push/slide paper** - Contact-assisted sliding

## Optimized Configuration

### Real-World Friction Coefficients:

| Material Pair | Typical μ | Reference |
|--------------|-----------|-----------|
| Paper on wood/plastic table | **0.15-0.25** | Engineering handbooks |
| Rubber fingertips on paper | **1.2-2.0** | Gripper design literature |
| Steel on steel | 0.7-0.8 | (for comparison) |

### Proposed Settings:

```xml
<!-- Paper: Lower friction for easy sliding on table -->
<geom name="paper_geom" friction="0.15 0.001 0.00005"/>

<!-- Table: Keep low for smooth paper motion -->
<geom name="table_surface" friction="0.20 0.001 0.00005"/>

<!-- Fingertips: HIGH friction for firm grip -->
<geom name="fixed_fingertip" friction="2.0 0.003 0.0001"/>
<geom name="moving_fingertip" friction="2.0 0.003 0.0001"/>
```

### Expected Results:

| Surface Pair | New Friction | Improvement |
|--------------|--------------|-------------|
| Paper-Table | 0.15 × 0.20 = **0.03 effective** | 20× easier to slide |
| Fingertip-Paper | 2.0 × 0.15 = **0.30 effective** | Still allows sliding |
| Gripping force | High μ=2.0 | Prevents slipping when grasping |

## MuJoCo Friction Model

MuJoCo uses 3 friction coefficients: `[slide, spin, roll]`

- **slide (μ_slide)**: Tangential friction (most important)
- **spin**: Torsional friction about contact normal (low for paper)
- **roll**: Rolling resistance (minimal for sliding objects)

**Effective friction between two geoms:**
- MuJoCo uses a **geometric mean** or **min** depending on solver
- Safe assumption: `μ_eff ≈ min(μ1, μ2)` for slide friction
- This is why we need HIGH fingertip friction (2.0) to dominate paper friction

## Implementation Strategy

1. **Reduce paper slide friction**: 0.60 → 0.15
2. **Reduce table friction**: 0.30 → 0.20 (already low, but consistency)
3. **Increase fingertip friction**: 1.0 → 2.0

This creates a **friction hierarchy**:
- **Fingertip-Paper**: HIGH (μ=2.0 × 0.15 = strong grip)
- **Paper-Table**: LOW (μ=0.15 × 0.20 = easy slide)

## Testing Plan

After changes, test in interactive viewer:

1. **Paper sliding**: Paper should slide easily on table with small force
2. **Fingertip grip**: Fingertips should not slip when pushing paper
3. **Combined motion**: Fingertips can push paper across table smoothly

## References

- MuJoCo documentation: Contact dynamics and friction models
- Robotics gripper design: Rubber friction coefficients 1.2-2.5
- Material science: Paper-wood friction typically 0.15-0.25
