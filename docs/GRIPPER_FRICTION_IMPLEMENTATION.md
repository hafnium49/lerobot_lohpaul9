# High-Friction Gripper Tips Implementation

**Date:** 2025-10-16
**Status:** ✅ COMPLETE - Phase A Ready (Pushing-Only Mode)

---

## Summary

Successfully implemented high-friction gripper fingertips for the SO-101 robot in MuJoCo simulation, simulating nitrile gloves over work gloves. The implementation includes:

- Capsule-based fingertip geometry (broader contact patch than spheres)
- Physics best practices (proper friction separation, contact margin, collision filtering)
- Runtime friction modulation (release hack to prevent "glued paper")
- Domain randomization for sim-to-real transfer
- Staged curriculum support (Phase A: pushing-only → Phase B: grasping enabled)

---

## Implementation Details

### 1. XML Modifications (`paper_square_realistic.xml`)

#### A. Contact Margin (Line 35)
```xml
<default class="task_objects">
  <geom condim="3" solref="0.005 1" solimp="0.9 0.95 0.001"
        friction="1 0.1 0.1" margin="0.001"/>
</default>
```
**Purpose:** Prevents Z-fighting, improves solver stability

#### B. Gripper Material (Line 75)
```xml
<material name="gripper_rubber" rgba="0.3 0.3 0.3 1"/>
```
**Purpose:** Visual distinction (dark gray)

#### C. Table Physics (Line 81)
```xml
<geom name="table_surface" type="plane" size="1 1 0.1"
      material="matte_black"
      friction="0.30 0.001 0.00005"  <!-- High slide, low spin/roll -->
      conaffinity="2"  <!-- Matches paper/fingertips -->
      class="task_objects"/>
```

#### D. Paper Physics (Lines 201-209)
```xml
<geom name="paper_geom" type="box"
      size="0.105 0.1485 0.0005"  <!-- 0.5mm half-thickness (solver-friendly) -->
      friction="0.60 0.001 0.00005"  <!-- Reduced spin/roll -->
      contype="1" conaffinity="2"  <!-- Collision filtering -->
      class="task_objects"/>
```
**Change:** Increased thickness from 0.15mm → 0.5mm for stability

#### E. Tape Visual-Only (Lines 225-233)
```xml
<geom name="tape_contact" ... contype="0" conaffinity="0"/>
```
**Purpose:** Remove unnecessary collision checks

#### F. Fixed Jaw Fingertip (Lines 172-179)
```xml
<geom name="fixed_fingertip" type="capsule"
      class="task_objects"
      fromto="-0.008 -0.002 -0.095   -0.004 -0.002 -0.095"  <!-- 4mm length -->
      size="0.004"  <!-- 4mm radius -->
      material="gripper_rubber"
      friction="1.0 0.003 0.0001"  <!-- High slide, tiny spin/roll -->
      contype="0" conaffinity="2"/>  <!-- Phase A: DISABLED -->
```

#### G. Moving Jaw Fingertip (Lines 208-215)
```xml
<geom name="moving_fingertip" type="capsule"
      class="task_objects"
      fromto="0.008 -0.030 0.015    0.004 -0.030 0.015"
      size="0.004"
      material="gripper_rubber"
      friction="1.0 0.003 0.0001"
      contype="0" conaffinity="2"/>  <!-- Phase A: DISABLED -->
```

**Key Design Choices:**
- **Capsules** (not spheres): Broader contact patch, less chatter
- **High slide friction** (1.0): Simulates nitrile+glove
- **Low spin/roll** (0.003, 0.0001): Realistic contact mechanics
- **Phase A disabled** (`contype="0"`): Pushing-only curriculum

---

### 2. Python Environment Modifications (`so101_residual_env.py`)

#### A. Fingertip Tracking (Lines 138-152)
```python
# Gripper fingertip IDs for runtime friction control (release hack)
self.fixed_fingertip_id = None
self.moving_fingertip_id = None
self.gripper_base_friction = 1.0  # Normal grasping friction
self.gripper_release_friction = 0.6  # Lowered during opening
self.gripper_release_threshold = 0.5  # Gripper position threshold

# Try to get fingertip geom IDs
try:
    self.fixed_fingertip_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "fixed_fingertip")
    self.moving_fingertip_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "moving_fingertip")
    if self.fixed_fingertip_id >= 0 and self.moving_fingertip_id >= 0:
        print("✅ Gripper fingertips found - release control enabled")
except:
    pass  # Fingertips not available, no problem
```

#### B. Friction Modulation Method (Lines 225-249)
```python
def _update_gripper_friction(self):
    """
    Modulate fingertip friction based on gripper state (release hack).

    When gripper is opening (releasing paper), temporarily lower
    friction to prevent "glued paper" syndrome. This simulates the
    natural release behavior of nitrile gloves.
    """
    if self.fixed_fingertip_id is None or self.moving_fingertip_id is None:
        return
    if self.fixed_fingertip_id < 0 or self.moving_fingertip_id < 0:
        return

    # Get current gripper position (joint 5)
    gripper_pos = self.data.qpos[self.joint_ids[5]]

    # If gripper is opening (pos > threshold), use release friction
    if gripper_pos > self.gripper_release_threshold:
        target_friction = self.gripper_release_friction
    else:
        target_friction = self.gripper_base_friction

    # Update friction coefficients (only slide component, keep spin/roll)
    self.model.geom_friction[self.fixed_fingertip_id, 0] = target_friction
    self.model.geom_friction[self.moving_fingertip_id, 0] = target_friction
```

#### C. Step Integration (Lines 437-438)
```python
# Update gripper friction based on state (release hack)
self._update_gripper_friction()
```

#### D. Friction Randomization in Reset (Lines 377-383)
```python
# Randomize gripper friction (simulates dust/humidity on nitrile gloves)
if self.randomize and self.fixed_fingertip_id is not None:
    if self.fixed_fingertip_id >= 0 and self.moving_fingertip_id >= 0:
        # Vary base/release friction by ±15%
        friction_scale = self.np_random.uniform(0.85, 1.15)
        self.gripper_base_friction = 1.0 * friction_scale
        self.gripper_release_friction = 0.6 * friction_scale
```

---

## Validation Results

### Test Script: `scripts/test_gripper_friction.py`

**All tests passed:**
- ✅ Fingertip geoms found (IDs: 29, 32)
- ✅ Friction values correct (1.0, 0.003, 0.0001)
- ✅ Collision disabled for Phase A (contype=0)
- ✅ Friction randomization working (mean=0.987, std=0.092, range=[0.878, 1.130])
- ✅ Runtime friction modulation implemented (release hack ready)

**Note:** Release hack behavior not observable in Phase A since gripper doesn't physically interact with paper (collision disabled).

---

## Staged Curriculum Training

### Phase A: Pushing-Only (Current State)
**Status:** ✅ Ready to train

**Configuration:**
- Fingertip collision: **DISABLED** (`contype="0"`)
- Paper collision: Enabled (table friction only)
- Strategy: Robot must push/slide paper using gripper body

**Expected Behavior:**
- Gripper passes through paper (no contact forces)
- Robot learns pushing strategies
- Foundation for grasping behaviors

**Training Command:**
```bash
python scripts/train_ppo_residual.py --randomize
```

**Target Metrics:**
- 50k steps @ 30Hz ≈ 27 minutes wall time
- Success rate: 40-70% (pure pushing)

---

### Phase B: Grasping Enabled (Future)
**Status:** ⏸️ Awaiting Phase A completion

**Activation Steps:**
1. Edit `src/lerobot/envs/so101_assets/paper_square_realistic.xml`
2. Line ~172: Change `contype="0"` → `contype="2"` (fixed_fingertip)
3. Line ~208: Change `contype="0"` → `contype="2"` (moving_fingertip)
4. Save and reload environment

**Expected Behavior:**
- Gripper can grasp paper (high friction contact)
- Release hack prevents "glued paper"
- Robot learns grasping + placement strategies

**Training Command:**
```bash
# Continue from Phase A checkpoint
python scripts/train_ppo_residual.py --randomize --checkpoint outputs/phase_a/checkpoint.pt
```

**Target Metrics:**
- Additional 50k-100k steps
- Success rate: 85-90% (grasping + pushing)

---

## Physics Parameters Summary

| Component | Slide Friction | Spin Friction | Roll Friction | Contype | Conaffinity |
|-----------|---------------|---------------|---------------|---------|-------------|
| Table     | 0.30          | 0.001         | 0.00005       | 1       | 2           |
| Paper     | 0.60          | 0.001         | 0.00005       | 1       | 2           |
| Fingertips| 1.0 (base)    | 0.003         | 0.0001        | 0 → 2   | 2           |
| Tape      | N/A           | N/A           | N/A           | 0       | 0           |

**Contact Margin:** 0.001m (all task_objects)
**Paper Thickness:** 0.5mm (was 0.15mm)

---

## Domain Randomization

**Per Episode:**
1. **Paper pose:** Position ±5cm XY, orientation ±17°
2. **Paper friction:** ±20% variation (μ = 0.48-0.72)
3. **Gripper friction:** ±15% variation (μ = 0.85-1.15)

**Purpose:** Improve sim-to-real transfer robustness

---

## Testing & Validation

### Visual Inspection
```bash
python scripts/view_world.py
```

**What to check:**
1. Dark gray capsules at gripper tips (~95mm below gripper frame)
2. Press `Ctrl+F` to enable contact forces visualization
3. Close gripper and observe forces (Phase B only)

### Environment Smoke Test
```bash
python scripts/test_gripper_friction.py
```

**Validates:**
- Fingertip geoms present
- Friction values correct
- Randomization working
- Release hack implemented

### Full Environment Test
```bash
python -m lerobot.envs.so101_residual_env
```

**Validates:**
- Environment loads
- Reset works
- Step works
- Reward computation correct

---

## Known Limitations

1. **Phase A gripper behavior:** In current state (contype=0), gripper cannot affect paper via fingertips. This is intentional for curriculum learning.

2. **Release hack not observable in Phase A:** Since fingertips don't collide, the friction modulation has no visible effect until Phase B.

3. **Sim-to-real gap:** MuJoCo friction model is simplified. Real nitrile gloves have:
   - Anisotropic friction (direction-dependent)
   - Adhesion effects (not modeled)
   - Wear and contamination (approximated by randomization)

4. **Gripper actuation:** SO-101 has coupled gripper fingers (single actuator). This is correctly modeled but limits grasping strategies.

---

## Future Work

### Immediate (Phase B)
- Enable fingertip collision after Phase A converges
- Tune release threshold based on training behavior
- Add gripper force sensors if hardware available

### Long-term (Phase 8)
- Vision-based residual policy (replace privileged state)
- Domain randomization for lighting, camera pose, table texture
- Sim-to-real validation on physical SO-101 hardware
- GR00T base policy integration (once Isaac-GR00T setup)

---

## References

**Related Documents:**
- [GROOT_INTEGRATION.md](./GROOT_INTEGRATION.md) - GR00T base policy integration
- [SO101_SETUP.md](../src/lerobot/robots/so101_mujoco/README.md) - Robot description

**Technical Resources:**
- MuJoCo Contact Documentation: https://mujoco.readthedocs.io/en/stable/modeling.html#geom
- Friction Cone Model: https://mujoco.readthedocs.io/en/stable/computation.html#cocontact

---

**Last Updated:** 2025-10-16
**Status:** ✅ Complete - Ready for Phase A Training
**Next Milestone:** Run 50k step training with pushing-only curriculum
