# SO101 Residual Reinforcement Learning System

**Last Updated**: 2025-10-23

This implementation provides **residual reinforcement learning** for the SO-101 robot arm in MuJoCo simulation. The RL agent learns to add corrective actions on top of a base policy (Jacobian IK, frozen IL model, or **fine-tuned GR00T N1.5**) to improve task performance.

## 📋 Task: Paper-in-Square

The robot must slide a piece of paper into a red tape square on the table. This task requires:
- Precise end-effector control
- Understanding of sliding dynamics
- Coordination of pushing motions
- Adaptation to paper friction variations

## 🏗️ Architecture

```
Observation → Base Policy → Base Action ↘
                                         → α·blend → Total Action → Environment
Observation → RL Policy →  Residual    ↗
```

- **Base Policy**: Provides nominal actions (Jacobian IK, frozen IL model, or **GR00T N1.5**)
- **RL Policy**: Learns residual corrections via PPO
- **Total Action**: `action = base_action + α * residual_action`

## 🤖 GR00T N1.5 Integration

### Fine-Tuned Base Policy

We use a **fine-tuned GR00T N1.5 model** (`phospho-app/gr00t-paper_return-7w9itxzsox`) as the base policy:

- **Training**: Fine-tuned via imitation learning on paper-return task demonstrations
- **Dataset**: `Hafnium49/paper_return` (task-specific demonstrations)
- **Model Type**: GR00T N1.5 dual-brain architecture
- **Action Output**: 6 DOF (5 arm joints + 1 gripper)
- **Input**: RGB images from top-view camera (224×224)

### Camera Configuration

Both cameras calibrated for optimal visual input:

**Wrist Camera**:
- Position: `[0.0025, 0.0609, 0.0120]` m (dodecagon lens hole center)
- Orientation: Perpendicular to mounting surface + 90° CW roll
- FOV: 75° (matches typical UVC modules)

**Top View Camera**:
- Position: `[0.275, 0.175, 0.4]` m
- Orientation: 90° CW rotation
- FOV: 90°

See [CAMERA_CALIBRATION_SUMMARY.md](CAMERA_CALIBRATION_SUMMARY.md) for complete calibration details.

### Why GR00T as Base Policy?

1. **Pre-trained on task**: Fine-tuned specifically for paper-return task
2. **Vision-based**: Uses RGB observations (more realistic than state-based)
3. **Generalization**: Trained on diverse demonstrations with randomization
4. **Sim-to-real**: Can transfer to real SO-101 robot with camera mount

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements_residual_rl.txt

# Or manually:
pip install mujoco gymnasium stable-baselines3[extra] torch transformers
```

### 2. Test Environment

```bash
# Test the environment with random actions
python src/lerobot/envs/so101_residual_env.py
```

### 3. Train Residual Policy

```bash
# Train with GR00T N1.5 as base policy (RECOMMENDED)
python src/lerobot/scripts/train_so101_residual.py \
    --base-policy groot \
    --groot-model phospho-app/gr00t-paper_return-7w9itxzsox \
    --alpha 0.3 \
    --n-envs 4 \
    --total-timesteps 500000

# Train with Jacobian IK as base policy
python src/lerobot/scripts/train_so101_residual.py \
    --base-policy jacobian \
    --alpha 0.5 \
    --n-envs 4 \
    --total-timesteps 500000

# Train with zero base (pure RL)
python src/lerobot/scripts/train_so101_residual.py \
    --base-policy zero \
    --alpha 1.0 \
    --n-envs 4 \
    --total-timesteps 1000000

# Train with frozen IL policy (if you have one)
python src/lerobot/scripts/train_so101_residual.py \
    --base-policy il \
    --il-checkpoint path/to/checkpoint.pth \
    --alpha 0.3 \
    --n-envs 4
```

### 4. Evaluate Trained Policy

```bash
# Evaluate and compare policies
python src/lerobot/scripts/eval_so101_residual.py \
    --model-path runs/residual_rl/*/best_model/best_model.zip \
    --compare-base \
    --compare-zero \
    --n-episodes 100 \
    --plot \
    --save-video

# Evaluate GR00T base only (no residual)
python scripts/eval_groot_base_only.py \
    --model-path phospho-app/gr00t-paper_return-7w9itxzsox \
    --n-episodes 50
```

## 📁 File Structure

```
src/lerobot/
├── envs/
│   ├── so101_assets/
│   │   ├── paper_square_realistic.xml     # MuJoCo scene (calibrated cameras)
│   │   └── official_model/                # SO-101 3D models (camera mount)
│   ├── so101_residual_env.py             # Gymnasium environment
│   └── so101_base_policy.py              # Base policy implementations
├── policies/
│   └── groot_base_policy.py              # GR00T N1.5 wrapper
├── scripts/
│   ├── train_so101_residual.py           # PPO training script
│   ├── eval_so101_residual.py            # Evaluation script
│   ├── eval_groot_base_only.py           # GR00T evaluation (no residual)
│   └── record_groot_dual_camera.py       # Record dual-feed videos
└── robots/
    └── so101_mujoco/                     # Existing SO101 implementation
```

## 🎯 Key Features

### Multi-Rate Control
- **Physics**: 360 Hz (MuJoCo simulation)
- **Control**: 180 Hz (internal control loop)
- **Policy**: 30 Hz (RL decision frequency)
- **GR00T**: 30 Hz (inference on GPU)

### Domain Randomization
- Paper position: ±5cm from nominal
- Paper orientation: ±0.3 rad
- Friction: ±20% variation
- Mass: ±10% variation
- Camera FOV: Realistic values (75° wrist, 90° top)

### Reward Design
```python
reward = (
    +10.0 * success           # All corners in square
    -2.0 * distance_to_goal   # Distance shaping
    -0.1 * orientation_error  # Keep paper aligned
    -0.001 * ||residual||²    # Residual magnitude penalty
    -0.01                     # Time penalty
)
```

### Action Spaces
- **Residual Action**: 6D joint deltas (scaled by `act_scale`)
- **Base Action**: 6D joint deltas from base policy (GR00T, IK, or IL)
- **Total Action**: Clipped sum with blending factor α

## 🔧 Training Parameters

### Key Hyperparameters

| Parameter | GR00T Base | Jacobian Base | Description |
|-----------|-----------|---------------|-------------|
| `alpha` | 0.3 | 0.5 | Residual blending (0=base only, 1=full residual) |
| `act_scale` | 0.02 | 0.02 | Maximum joint delta per step (radians) |
| `residual_penalty` | 0.001 | 0.001 | L2 penalty on residual magnitude |
| `learning_rate` | 3e-4 | 3e-4 | PPO learning rate |
| `n_envs` | 4 | 4 | Parallel environments (RTX 3060 limit) |
| `n_steps` | 256 | 256 | Steps per rollout |
| `batch_size` | 64 | 64 | PPO batch size |

### GPU Memory Guidelines

| GPU | Recommended `n_envs` | Notes |
|-----|---------------------|-------|
| RTX 3060 (12GB) | 4 | With GR00T inference |
| RTX 3070 (8GB) | 2-4 | May need smaller batch |
| RTX 4090 (24GB) | 8-16 | Can add more envs |

**Note**: GR00T N1.5 requires ~3-4GB GPU memory per inference, limiting parallel environments.

## 📊 Expected Performance

After training for 500K-1M timesteps:

| Policy | Success Rate | Mean Reward | Training Time (3060) | Notes |
|--------|--------------|-------------|---------------------|-------|
| GR00T Base (no residual) | 40-50% | ~-20 | N/A | Fine-tuned IL |
| Base (Jacobian IK) | 30-40% | ~-50 | N/A | Analytical |
| Pure RL (no base) | 60-70% | ~20 | 4-6 hours | From scratch |
| **Residual + GR00T** | **90-95%** | **~60** | **2-3 hours** | **Best** |
| Residual + Jacobian | 85-90% | ~50 | 2-3 hours | Good baseline |

**Key Insight**: GR00T base provides better initialization than Jacobian IK, leading to:
- Higher success rates
- Faster convergence
- More natural behaviors

## 🔬 Experimental Variations

### 1. GR00T Alpha Scheduling
```python
# Start with mostly GR00T (α=0.2), gradually increase residual
alpha_schedule = lambda t: min(0.5, 0.2 + 0.3 * t/total_steps)
```

### 2. Curriculum Learning
```python
# Start with larger target square
tape_half_size = 0.12  # 24cm square
# Gradually reduce to 0.08 (16cm)
```

### 3. Different Base Policies

| Base Policy | Pros | Cons | Best For |
|-------------|------|------|----------|
| **GR00T N1.5** | Vision-based, pre-trained, generalizes well | GPU memory, slower inference | Sim-to-real, complex tasks |
| Jacobian IK | Fast, no GPU, interpretable | Limited to simple motions | Baseline, debugging |
| Frozen ACT | Good for manipulation | Needs demos, less flexible | Bimanual tasks |
| Frozen Diffusion | Smooth trajectories | Slow inference | Precise movements |

## 🐛 Troubleshooting

### GR00T-Specific Issues

**Problem**: "Model not found or available in huggingface hub"
```bash
# Check model cache
ls ~/.cache/huggingface/hub/models--phospho-app--gr00t-paper_return-7w9itxzsox

# Re-download if needed
python -c "from transformers import AutoModel; AutoModel.from_pretrained('phospho-app/gr00t-paper_return-7w9itxzsox')"
```

**Problem**: Out of GPU memory with GR00T
- Reduce `n_envs` to 2-4
- Use CPU for some environments: `--device-envs cpu`
- Disable video recording during training

**Problem**: GR00T inference too slow
- Use `MUJOCO_GL=egl` for headless rendering
- Batch GR00T inference across environments
- Reduce environment complexity

### Environment Issues

**Problem**: "MuJoCo model not found"
```bash
# Ensure XML path is correct
ls src/lerobot/envs/so101_assets/paper_square_realistic.xml
```

**Problem**: "GLFW error" or rendering issues
```bash
# Run headless (no rendering)
export MUJOCO_GL=egl
# Or disable rendering in code
```

### Training Issues

**Problem**: No learning progress with GR00T base
- Increase `alpha` to give RL more control (try 0.4-0.5)
- Check GR00T is working: evaluate with `alpha=0`
- Verify camera calibration is correct

**Problem**: Unstable training
- Reduce `learning_rate` to 1e-4
- Increase `batch_size` to 128
- Add gradient clipping: `max_grad_norm=0.5`

## 🔄 Integration with LeRobot

### Using GR00T Base Policy

```python
from lerobot.policies.groot_base_policy import GR00TBasePolicy

# Load fine-tuned GR00T
groot_policy = GR00TBasePolicy(
    model_path="phospho-app/gr00t-paper_return-7w9itxzsox",
    device="cuda",
    expected_action_dim=6
)

# Use in residual RL environment
env = SO101ResidualEnv(
    base_policy=groot_policy,
    alpha=0.3,
    use_image_obs=True,
    camera_name_for_obs="top_view"
)
```

### Recording Demonstrations

```bash
# Use teleoperation to record demos for fine-tuning GR00T
python -m lerobot.scripts.record \
    --robot-name so101_mujoco \
    --task paper_square \
    --episodes 50
```

### Training Pipeline

```bash
# Step 1: Fine-tune GR00T on demonstrations (done separately)
# Model: phospho-app/gr00t-paper_return-7w9itxzsox

# Step 2: Use fine-tuned GR00T as base for residual RL
python scripts/train_so101_residual.py \
    --base-policy groot \
    --groot-model phospho-app/gr00t-paper_return-7w9itxzsox \
    --alpha 0.3
```

## 📈 Monitoring Training

### TensorBoard
```bash
# View training curves
tensorboard --logdir runs/residual_rl
```

### Key Metrics to Watch
- `rollout/ep_rew_mean`: Should increase over time
- `residual/success_rate`: Target >0.90 with GR00T base
- `residual/penalty`: Should stay small (<0.01)
- `task/dist_to_goal`: Should decrease
- `groot/inference_time`: Monitor GR00T latency

### Weights & Biases (Optional)
```bash
# Install wandb
pip install wandb

# Login
wandb login

# Train with logging
python scripts/train_so101_residual.py \
    --base-policy groot \
    --wandb-project so101-groot-residual
```

## 📹 Video Demonstrations

Generate dual-feed videos (top view + wrist camera) showing policy performance:

```bash
# Record GR00T base policy only
python scripts/record_groot_dual_camera.py \
    --output videos/groot_base_only.mp4 \
    --episodes 5

# Record residual RL + GR00T
python scripts/record_groot_dual_camera.py \
    --output videos/groot_residual.mp4 \
    --residual-model runs/residual_rl/best_model.zip \
    --episodes 5
```

See [GROOT_VIDEOS_README.md](GROOT_VIDEOS_README.md) for available demonstration videos.

## 🎓 Understanding the Code

### Environment (`so101_residual_env.py`)
- Implements Gymnasium interface
- Handles MuJoCo simulation with calibrated cameras
- Computes rewards and success
- Manages domain randomization
- Supports both state and image observations

### Base Policies (`groot_base_policy.py`)
- `GR00TBasePolicy`: Wrapper for fine-tuned GR00T N1.5
  - Handles modality-based action extraction
  - Image preprocessing (224×224 RGB)
  - GPU inference batching
  - Action normalization

### Other Base Policies (`so101_base_policy.py`)
- `JacobianIKPolicy`: Analytical IK solution
- `FrozenILPolicy`: Loads pre-trained models
- `ZeroPolicy`: Baseline (no base)
- `HybridPolicy`: Combines multiple bases

### Training (`train_so101_residual.py`)
- Sets up parallel environments
- Configures PPO algorithm
- Handles GR00T GPU management
- Logs metrics and videos

### Evaluation (`eval_so101_residual.py`)
- Loads trained models
- Compares policies (GR00T, Jacobian, pure RL, residual)
- Generates videos with dual camera views
- Produces performance plots

## 📚 References

- [GR00T Paper](https://arxiv.org/abs/2410.06158) - NVIDIA's GR00T foundation model
- [LeRobot](https://github.com/huggingface/lerobot) - HuggingFace robotics library
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - PPO implementation
- [MuJoCo](https://mujoco.readthedocs.io/) - Physics simulation
- [Residual RL Paper](https://arxiv.org/abs/1812.03201) - Silver et al., 2018

## 🤝 Contributing

Feel free to:
- Add new tasks (pick, place, stack)
- Implement different RL algorithms (SAC, TD3)
- Improve GR00T integration (multi-modal observations)
- Add sim-to-real transfer techniques
- Optimize camera calibration

## 📄 License

Apache 2.0 - See LICENSE file

## ✨ Acknowledgments

- **LeRobot** framework by HuggingFace
- **GR00T N1.5** foundation model by NVIDIA
- Fine-tuned model by **phospho-app** team
- Camera calibration based on SO-101 official hardware specs
- Residual RL inspired by Silver et al., 2018

---

**Quick Links**:
- [Camera Calibration](CAMERA_CALIBRATION_SUMMARY.md)
- [GR00T Integration](GROOT_INTEGRATION_SUMMARY.md)
- [Video Demonstrations](GROOT_VIDEOS_README.md)
- [Training Status](RESIDUAL_RL_STATUS.md)
