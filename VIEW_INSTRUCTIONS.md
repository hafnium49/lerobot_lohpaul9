# Viewing the SO-101 Paper-Square MuJoCo World

You have **3 easy ways** to visualize your world:

---

## Method 1: Python Script (Recommended)

**Simple and pre-configured with nice initial pose:**

```bash
python view_world.py
```

This script:
- ✅ Sets the robot to a good viewing pose
- ✅ Positions the paper between robot and target
- ✅ Shows helpful info about the world
- ✅ Lists all viewer controls

---

## Method 2: One-Liner (Fastest)

**No script needed, just point to your XML:**

```bash
python -m mujoco.viewer --mjcf=src/lerobot/envs/so101_assets/paper_square.xml
```

This launches the MuJoCo viewer directly from the command line.

---

## Method 3: Drag-and-Drop (Most Flexible)

**Launch empty viewer, then drag your file:**

```bash
# Launch empty viewer
python -m mujoco.viewer
```

Then **drag and drop** the XML file (`src/lerobot/envs/so101_assets/paper_square.xml`) into the viewer window.

---

## Viewer Controls

Once the viewer is open:

| Action | Control |
|--------|---------|
| **Rotate camera** | Mouse drag |
| **Zoom in/out** | Scroll wheel |
| **Pan camera** | Right-click drag |
| **Select body** | Double-click on body |
| **Apply force** | Ctrl + Right-click on body |
| **Pause/Resume** | Space bar |
| **Reset simulation** | Backspace |
| **Toggle left panel** | Tab |
| **Toggle right panel** | Shift+Tab |
| **Simulation speed** | [ and ] keys |
| **Exit** | ESC or close window |

---

## Camera Views

In the right panel (Shift+Tab), you can select different cameras:

- **Free camera** - User-controlled (default)
- **Top view** - Bird's eye view of the table
- **Side view** - Profile view of the robot

---

## Troubleshooting

### "Cannot connect to display"
If you're on a headless server or WSL without display:

```bash
export MUJOCO_GL=egl
python view_world.py
```

### macOS Issues
On macOS, MuJoCo requires the main thread for graphics:

```bash
mjpython view_world.py
```

### Viewer window is black
Try toggling rendering modes in the right panel (Shift+Tab) under "Rendering flags"

---

## What You Should See

Your world contains:

1. **Gray table/floor** - The workspace surface
2. **Dark blue robot base** - SO-101 arm mounted at (0, -0.35, 0)
3. **Gray robot arm** - 5 DOF arm + gripper
4. **White rectangle** - A5 paper (148×210mm) at ~(0.30, 0, 0)
5. **Red dashed square** - Target zone (160×160mm) at (0.55, 0, 0)
6. **Green dot** - End-effector site (gripper tip)

**Task:** Push the paper into the red target square!

---

## Next Steps

After viewing the world:

1. **Check physics** - Watch how the paper slides when you apply forces
2. **Test friction** - Does the paper slide realistically?
3. **Adjust if needed** - Edit `paper_square.xml` friction values
4. **Start training** - Once satisfied, run residual RL training

---

## Quick Test

To test that everything works:

```bash
# Activate environment
source .venv/bin/activate

# Launch viewer
python view_world.py
```

You should see the robot, paper, and target square. The paper should be 25cm away from the target (orange arrow in schematic).
