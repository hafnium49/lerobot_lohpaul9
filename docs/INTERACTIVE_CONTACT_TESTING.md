# Interactive Contact Testing with MuJoCo Viewer

## Overview

The `view_world_with_contacts.py` script launches an interactive MuJoCo viewer with **real-time contact detection feedback**. This allows you to manually manipulate the SO-101 robot and see exactly when fingertip-paper, robot-paper, and robot-table contacts occur.

## Usage

```bash
python view_world_with_contacts.py
```

## Features

### Real-Time Contact Detection

As you manipulate the robot in the viewer, the terminal will show color-coded contact events:

- **🟢 GREEN**: Fingertip-paper contact (GOOD! +0.1 reward)
- **🟡 YELLOW**: Robot-paper contact (WARNING - -0.2 penalty)
- **🔴 RED**: Robot-table contact (BAD - -0.5 penalty)

### How to Manipulate the Robot

1. **Launch the viewer:**
   ```bash
   python view_world_with_contacts.py
   ```

2. **Move robot parts:**
   - **Ctrl + Right-click** on any robot body (gripper, wrist, arm, etc.)
   - **Drag** to apply force and move that body
   - The physics will naturally move the robot

3. **Test fingertip-paper contact:**
   - Ctrl+Right-click on the gripper body
   - Drag it toward the white paper on the table
   - Watch the terminal for green contact messages!

4. **Test robot-paper contact:**
   - Ctrl+Right-click on the wrist or arm
   - Drag it into the paper
   - Yellow warning will appear in terminal

5. **Test robot-table contact:**
   - Ctrl+Right-click on the arm
   - Drag it down toward the table
   - Red alert will appear in terminal

## Viewer Controls

| Action | Control |
|--------|---------|
| Rotate camera | Left-click + drag |
| Zoom | Mouse scroll |
| Pan camera | Right-click + drag |
| Select body | Double-click |
| **Apply force** | **Ctrl + Right-click + drag** |
| Pause/Resume | Space |
| Toggle left panel | Tab |
| Toggle right panel | Shift + Tab |
| Reset simulation | Backspace |
| Exit | ESC or close window |

## Example Terminal Output

When you successfully touch the paper with fingertips:

```
================================================================================
🟢 FINGERTIP-PAPER CONTACT DETECTED! (Reward: +0.1)
================================================================================
   fixed_fingertip <-> paper_geom
================================================================================
```

When contact ends:
```
✓ Fingertip-paper contact ended
```

## Tips for Testing

1. **Start with fingertip contact:**
   - The grey spheres (16mm diameter) at gripper tips are the fingertips
   - Move the gripper toward the white A5 paper
   - Contact should be easy to trigger with 16mm spheres on 210mm×297mm surface

2. **Check contact frequency:**
   - Contacts are checked every 10 physics steps to reduce terminal spam
   - If contact is very brief, it may not be printed

3. **Verify all three contact types:**
   - Fingertip-paper (green) - Move gripper to paper
   - Robot-paper (yellow) - Move arm/wrist to paper
   - Robot-table (red) - Move arm down to table

4. **Physics is realistic:**
   - Robot collision prevents table penetration
   - You can push paper around with the gripper
   - Fingertips have high friction (μ=1.0) for gripping

## Technical Details

### Contact Detection Logic

The script uses the same contact detection logic as `so101_residual_env.py`:

1. **Geom ID tracking:**
   - 13 robot collision geoms (shoulder, upper_arm, lower_arm, wrist, gripper, jaw)
   - 2 fingertip geoms (fixed and moving)
   - 1 table surface geom
   - 1 paper body

2. **Contact buffer iteration:**
   - Checks `data.ncon` contacts every frame
   - Matches geom pairs against tracked IDs
   - Classifies into three contact types

3. **Event detection:**
   - Only prints when contact state changes (new contact or contact ended)
   - Prevents terminal spam
   - Shows contact details (which geoms are touching)

### Collision Settings

All properly configured:
- Fingertips: `contype=2, conaffinity=2, radius=8mm`
- Paper: `contype=1, conaffinity=2, size=210×297×1mm`
- Robot arm: `contype=1, conaffinity=2`
- Table: `contype=2, conaffinity=2`

Collision formula: `(geom1.contype & geom2.conaffinity) || (geom2.contype & geom1.conaffinity)`

## Troubleshooting

### Viewer won't launch

**On headless systems:**
```bash
export MUJOCO_GL=egl
python view_world_with_contacts.py
```

**On macOS:**
```bash
mjpython view_world_with_contacts.py
```

### No contacts detected

1. **Check you're using Ctrl+Right-click** to apply forces (not just dragging)
2. **Move gripper close to paper** - try to intersect the grey fingertip spheres with the white paper
3. **Check terminal output** - contacts print in color
4. **Contact may be brief** - hold Ctrl+Right-click for sustained contact

### Contacts spam terminal

This is normal! It means contact detection is working. Each contact state change triggers a message.

## Comparison to Automated Tests

| Test Type | Pros | Cons |
|-----------|------|------|
| **Interactive Viewer** | See contacts happen in real-time, intuitive manipulation, visual feedback | Manual, requires display, subjective |
| **Automated Tests** | Reproducible, quantitative, no display needed | Can't see what's happening, harder to debug |

**Recommendation:** Use both!
- Interactive viewer to **verify** contacts work
- Automated tests for **CI/CD** and benchmarking

---

**Created:** 2025-10-21
**Purpose:** Verify fingertip-paper contact detection works interactively
**Status:** Ready to use
