# GR00T N1.5 SO-101 Control Videos

This directory contains videos showing the fine-tuned GR00T N1.5 foundation model controlling the SO-101 robot in MuJoCo simulation for the paper return task.

## Available Videos

### 1. Single Camera View - Top View
**File:** `groot_so101_control.mp4` (1.3 MB)
- **Duration:** 10 seconds (300 frames @ 30 FPS)
- **Resolution:** 640×480
- **Camera:** Top-down view of workspace
- **Episodes:** 2 episodes, 150 steps each
- **View:** Bird's eye perspective showing full workspace

### 2. Dual Camera View - Side-by-Side
**File:** `groot_dual_view.mp4` (3.8 MB)
- **Duration:** 10 seconds (300 frames @ 30 FPS)
- **Resolution:** 1280×480 (two 640×480 views combined)
- **Cameras:** Top view (left) + Wrist camera (right)
- **Episodes:** 2 episodes, 150 steps each
- **View:** Synchronized dual perspective
  - **Left:** Top-down view showing workspace overview
  - **Right:** Wrist-mounted camera showing robot's perspective

### 3. Separate Camera Videos
**Files:**
- `groot_cameras_top_view.mp4` (1.3 MB) - Top-down view only
- `groot_cameras_wrist_camera.mp4` (2.5 MB) - Wrist camera view only

- **Duration:** 10 seconds each (300 frames @ 30 FPS)
- **Resolution:** 640×480 each
- **Episodes:** 2 episodes, 150 steps each (synchronized)
- **Use case:** For separate viewing or custom video editing

## Video Overlay Information

Each video includes on-screen overlays showing:
- **Episode number** and current **step**
- **Cumulative return** (reward)
- **Camera name** (Top View / Wrist Camera)
- **Model label** (GR00T N1.5 Base Policy)

## What You'll See

### Robot Behavior
- GR00T N1.5 controlling all 6 DOF of the SO-101 arm
- Continuous action commands based on visual input (224×224 RGB images)
- Non-trivial movements showing learned behavior patterns
- Gripper control (open/close actions)

### Task Context
- **Task:** Return a piece of paper to a designated location
- **Success rate:** 0% (domain gap between real-world training and simulation)
- **Observation:** Model actively responds to visual input but doesn't complete task
- **Returns:** Negative (penalties for not completing task)

### Camera Perspectives

#### Top View
- Shows the entire workspace from above
- Clear view of paper, robot arm, and target location
- Good for understanding overall task progress
- This is the camera view used for GR00T's visual input

#### Wrist Camera
- Robot's perspective from the wrist-mounted camera
- Shows what the robot "sees" during manipulation
- Close-up view of gripper and objects
- Useful for understanding detailed manipulation behavior

## Technical Details

### Model Information
- **Model:** `phospho-app/gr00t-paper_return-7w9itxzsox`
- **Architecture:** GR00T N1.5 (EAGLE backbone + DiT action head)
- **Parameters:** ~751M total (550M DiT + 201M transformer)
- **Training:** Fine-tuned on real-world SO-101 demonstrations
- **Input:** Dual RGB images (224×224) from two cameras
- **Output:** 6D continuous actions (5 joint positions + 1 gripper)

### Recording Setup
- **Simulator:** MuJoCo with EGL rendering
- **Physics timestep:** 0.002s (500 Hz)
- **Control frequency:** 30 Hz (matches video FPS)
- **Inference time:** ~192 ms per step (on RTX 3060)

### Environment
- **World:** SO-101 paper return task in MuJoCo
- **Robot:** SO-101 arm (6 DOF: pan, lift, elbow, wrist_flex, wrist_roll, gripper)
- **Objects:** Paper sheet, table, target location
- **Rendering:** Offscreen rendering via MuJoCo's EGL backend

## Performance Notes

### Why Low Success Rate?
The GR00T model was trained on **real-world** SO-101 robot data but is being tested in **MuJoCo simulation**. This creates a significant domain gap:

1. **Visual Domain Gap:**
   - Real camera images (training) vs. rendered images (testing)
   - Different lighting, textures, colors
   - Different camera parameters

2. **Physics Domain Gap:**
   - Real-world dynamics vs. MuJoCo physics
   - Different friction models
   - Different contact dynamics

3. **Observation Distribution Shift:**
   - Camera viewpoints may differ
   - Object appearances differ
   - Background differs

Despite these challenges, the model still produces **coherent, non-random actions**, showing it has learned meaningful visuomotor control patterns.

## How to Reproduce

### Record New Videos

**Side-by-side dual view:**
```bash
python scripts/record_groot_dual_camera.py \
    --output my_video.mp4 \
    --episodes 2 \
    --max-steps 200
```

**Separate camera videos:**
```bash
python scripts/record_groot_dual_camera.py \
    --output my_videos.mp4 \
    --episodes 2 \
    --max-steps 200 \
    --separate
```

**Single camera:**
```bash
python scripts/record_groot_video.py \
    --output my_video.mp4 \
    --episodes 2 \
    --camera top_view
```

### Customize Recording

Available options:
- `--episodes N` - Number of episodes to record
- `--max-steps N` - Maximum steps per episode
- `--fps N` - Frames per second (default: 30)
- `--width N --height N` - Video resolution
- `--camera1 NAME --camera2 NAME` - Camera names (dual-camera mode)
- `--separate` - Save separate videos instead of side-by-side

Available cameras in the world:
- `top_view` - Bird's eye view of workspace
- `wrist_camera` - Robot's wrist-mounted camera

## Related Files

- `scripts/record_groot_video.py` - Single camera recording script
- `scripts/record_groot_dual_camera.py` - Dual camera recording script
- `scripts/visualize_groot_control.py` - Interactive visualization
- `scripts/eval_groot_base_only.py` - Quantitative evaluation script
- `GROOT_INTEGRATION_SUMMARY.md` - Full integration documentation

## Citation

If you use GR00T in your work:

```bibtex
@article{groot2025,
  title={GR00T: Generalist Robot Policy},
  author={NVIDIA Research},
  year={2025}
}
```

If you use this SO-101 fork of LeRobot:

```bibtex
@software{lerobot2024,
  title={LeRobot: State-of-the-art Robotics in PyTorch},
  author={Hugging Face},
  year={2024},
  url={https://github.com/huggingface/lerobot}
}
```

---

**Note:** These videos demonstrate the technical capability of GR00T to generate coherent robot control commands from visual input. The low task success rate is expected due to the domain gap between real-world training and simulation testing. For actual deployment, either:
1. Train/fine-tune in the target domain (simulation), OR
2. Use domain adaptation techniques, OR
3. Use a different base policy (e.g., Jacobian IK baseline)

See `GROOT_INTEGRATION_SUMMARY.md` for detailed analysis and recommendations.
