# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeRobot is Hugging Face's state-of-the-art robotics library for real-world robotics in PyTorch. It provides models, datasets, and tools for imitation learning and reinforcement learning, with a focus on lowering the barrier to entry for robotics.

**Key Features:**
- Pre-trained models and datasets hosted on HuggingFace hub
- Support for real-world robots (SO-100/101, Koch, ALOHA, HopeJR, LeKiwi, etc.)
- Simulation environments (ALOHA, PushT, XArm)
- State-of-the-art policies (ACT, Diffusion, TDMPC, VQ-BeT, SmolVLA)
- Dataset collection and visualization tools
- Python 3.10+ and PyTorch 2.2+

**This Fork's Focus:**
This fork includes custom work on the SO-101 robot with MuJoCo simulation, featuring:
- Keyboard teleoperation for SO-101 in MuJoCo
- Jacobian-based XYZ control with manual wrist orientation
- Multi-rate control architecture (30Hz recording, 180Hz control, 360Hz physics)
- Direct wrist control improvements (recent commits)

## Development Setup

```bash
# Editable install
pip install -e .
pip install -e ".[all]"  # All features

# For contributors (preferred)
poetry sync --all-extras  # or: uv sync --all-extras
pre-commit install

# Environment variables
export HF_LEROBOT_HOME=~/.cache/huggingface/lerobot  # Dataset cache
export MUJOCO_GL=egl  # For headless rendering
```

## Common Commands

### Dataset Operations

```bash
# Visualize dataset
lerobot-dataset-viz --repo-id lerobot/pusht --episode-index 0

# Python usage
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset("lerobot/aloha_static_coffee")
# With temporal frames
dataset = LeRobotDataset("lerobot/pusht",
    delta_timestamps={"observation.image": [-1, -0.5, -0.2, 0]})
```

### Training and Evaluation

```bash
# Train policy
lerobot-train \
    dataset_repo_id=lerobot/aloha_sim_transfer_cube_human \
    policy=act \
    training.num_epochs=1000 \
    wandb.enable=true

# Evaluate policy
lerobot-eval \
    policy=path/to/pretrained_model \
    env_name=aloha
```

### Robot Operations

```bash
# Record dataset
lerobot-record --robot-name so101 --fps 30 --episodes 10 --dataset-name my_dataset

# Teleoperate robot
lerobot-teleoperate --robot-name so101

# Replay episodes
lerobot-replay --robot-name so101 --episodes 1,2,3

# Calibration and setup
lerobot-calibrate --robot-name so101
lerobot-find-cameras
lerobot-setup-motors --robot-name so101
```

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_control_robot.py

# End-to-end tests
make test-end-to-end

# Code quality checks
pre-commit run --all-files
```

## Architecture Overview

### High-Level Structure

```
lerobot/
├── src/lerobot/
│   ├── cameras/           # Camera interfaces (OpenCV, RealSense)
│   ├── configs/           # Configuration dataclasses and parsers
│   ├── datasets/          # Dataset loading, processing, and utilities
│   ├── envs/              # Simulation environments (wrappers)
│   ├── model/             # Neural network building blocks
│   ├── motors/            # Motor interfaces (Dynamixel, Feetech)
│   ├── optim/             # Optimizers and schedulers
│   ├── policies/          # Policy implementations (ACT, Diffusion, etc.)
│   ├── processor/         # Data pre/post-processors
│   ├── rl/                # RL utilities (SAC, TDMPC, WandB)
│   ├── robots/            # Robot interfaces and configurations
│   ├── scripts/           # CLI entry points (train, eval, record, etc.)
│   ├── teleoperators/     # Teleoperation interfaces (keyboard, gamepad, leader arms)
│   ├── templates/         # Configuration templates
│   ├── transport/         # gRPC for async inference
│   ├── utils/             # Common utilities
│   └── __init__.py        # Available components registry
├── tests/                 # Test suite
├── examples/              # Example scripts and tutorials
├── docs/                  # Documentation source
├── benchmarks/            # Performance benchmarks
└── docker/                # Docker configurations
```

### Core Components

#### 1. Datasets (`lerobot/datasets/`)

**Key Classes:**
- `LeRobotDataset`: Main dataset class with temporal indexing support
- `LeRobotDatasetMetadata`: Handles metadata (info, episodes, stats, tasks)
- `OnlineBuffer`: Real-time buffer for RL training

**Dataset Format:**
- Uses HuggingFace datasets (Arrow/Parquet backend)
- Videos stored as MP4 (torchcodec for decoding)
- Metadata in JSON/JSONL
- Features include observations (images, states), actions, episode info
- Support for delta timestamps for temporal sequences

**Key Files:**
- `lerobot_dataset.py`: Main dataset implementation
- `compute_stats.py`: Statistical computation for normalization
- `video_utils.py`: Video encoding/decoding
- `utils.py`: Dataset utilities and validation

#### 2. Policies (`lerobot/policies/`)

**Available Policies:**
- **ACT** (Action Chunking Transformer): For bimanual manipulation
- **Diffusion**: Diffusion policy for visuomotor control
- **TDMPC**: Temporal difference model predictive control
- **VQ-BeT**: Vector-quantized behavior transformer
- **SmolVLA**: Small vision-language-action model
- **Pi0/Pi0.5**: OpenVLA-based policies
- **SAC**: Soft actor-critic (RL baseline)

**Architecture:**
- `factory.py`: Policy factory with auto-detection
- Each policy has its own directory with config and implementation
- `pretrained.py`: Loading pretrained models from hub
- All policies inherit from base policy interface

#### 3. Robots (`lerobot/robots/`)

**Robot Interface (`robot.py`):**
```python
class Robot:
    def connect(self) -> None: ...
    def get_observation(self) -> dict: ...
    def send_action(self, action: dict) -> dict: ...
    def disconnect(self) -> None: ...
```

**Supported Robots:**
- SO-100/SO-101 (follower, leader, mujoco simulation)
- Koch (follower, leader)
- ALOHA
- HopeJR (humanoid arm)
- LeKiwi (mobile platform)
- Reachy2
- Stretch3
- ViperX

**Key Features:**
- Configuration via dataclasses
- Motor and camera abstraction
- Calibration utilities
- Safety checks and limits

#### 4. Teleoperators (`lerobot/teleoperators/`)

**Teleoperator Interface (`teleoperator.py`):**
```python
class Teleoperator:
    def get_action(self) -> dict: ...
    def is_recording(self) -> bool: ...
    def stop(self) -> None: ...
```

**Supported Interfaces:**
- Keyboard (WASD-style control)
- Gamepad (PS4/Xbox controllers)
- Leader arms (SO-100/101, Koch)
- Phone (mobile teleoperation)
- Homunculus (exoskeleton)

#### 5. Environments (`lerobot/envs/`)

**Simulation Environments:**
- ALOHA: Bimanual manipulation tasks
- PushT: 2D pushing task
- XArm: Pick and place tasks
- Libero: Long-horizon tasks

**Environment Wrappers:**
- Compatible with Gymnasium interface
- Automatic observation/action space handling
- Episode termination logic

#### 6. Configuration System (`lerobot/configs/`)

**Design:**
- Uses `draccus` for dataclass-based configs
- Hierarchical configuration with inheritance
- CLI argument parsing with dot notation
- Type-safe with validation

**Example:**
```python
from lerobot.configs.train import TrainPipelineConfig
config = TrainPipelineConfig(
    policy={"type": "act", "dim_model": 64},
    dataset={"repo_id": "lerobot/pusht"},
    batch_size=32
)
```

### Data Flow

#### Training Pipeline:
```
Dataset → DataLoader → Policy.forward() → Loss → Optimizer → Checkpoints
                ↓
         Normalization (stats from dataset)
                ↓
         Image transforms (augmentation)
```

#### Recording Pipeline:
```
Teleoperator.get_action() → Robot.send_action() → Robot.get_observation() → Dataset
                                      ↓
                          (Multi-rate control for smooth motion)
```

#### Evaluation Pipeline:
```
Env.reset() → Policy.select_action() → Env.step() → Metrics → Logs
                    ↓
         (Optional temporal window from history)
```

### Key Architectural Patterns

#### 1. Factory Pattern
All major components use factories:
- `make_policy()`, `make_dataset()`, `make_env()`
- Automatic registration via `__init__.py`
- Type-based instantiation

#### 2. Configuration-Driven
- Everything configurable via dataclasses
- Can override any parameter from CLI
- Configs saved with checkpoints for reproducibility

#### 3. HuggingFace Integration
- Datasets and models on HuggingFace hub
- Easy sharing and collaboration
- Version control via git revisions

#### 4. Temporal Indexing
- `delta_timestamps` for retrieving frame sequences
- Essential for policies that need observation history
- Efficient with pre-computed indices

#### 5. Multi-Rate Control (Custom Work)
- Recording at 30 Hz, control at 180 Hz, physics at 360 Hz
- Smooth integration of velocity commands
- Separates recording and replay modes

## Custom SO-101 MuJoCo Implementation

This fork includes a custom SO-101 robot implementation for MuJoCo simulation with keyboard teleoperation. Key innovations:

### Control Architecture
- **Jacobian-based XYZ control**: First 3 joints (pan, lift, elbow) control end-effector position
- **Manual wrist control**: Wrist flex and roll controlled directly by user (I/K and [/] keys)
- **Multi-rate execution**: 30Hz recording, 180Hz control, 360Hz physics for smooth motion

### Keyboard Controls
| Key | Action | Description |
|-----|--------|-------------|
| W/S | Forward/Backward | +Y / -Y world frame |
| A/D | Left/Right | -X / +X world frame |
| Q/E | Up/Down | +Z / -Z world frame |
| I/K | Wrist flex | Up/Down orientation |
| [/] | Wrist roll | CCW/CW rotation |
| O/C | Gripper | Open/Close |

### Files
- `src/lerobot/robots/so101_mujoco/` - Robot implementation
- `src/lerobot/teleoperators/keyboard/` - Keyboard teleoperator
- See `src/lerobot/robots/so101_mujoco/README.md` for detailed documentation

### Recent Changes
- Direct wrist control (PR #5)
- Block yaw control (PR #4)
- KeyboardEventManager improvements (PR #3)

## Code Standards

- **Formatter**: Ruff with 110 char line length
- **Linter**: Ruff with extensive rule set
- **Type Checking**: MyPy (gradually enabled)
- **Security**: Bandit for security checks
- **Docstrings**: Google style
- **Pre-commit**: Enforces all standards (`pre-commit run --all-files`)

### Adding New Components

**New Policy**: Add to `src/lerobot/policies/`, update `available_policies` in `__init__.py`
**New Robot**: Add to `src/lerobot/robots/`, implement Robot interface, update registry
**New Dataset**: Ensure LeRobotDataset compatible, update `available_datasets_per_env`

## Debugging Common Issues

**Dataset Issues:**
- Check `~/.cache/huggingface/lerobot/` for cached datasets
- Verify episode boundaries with `dataset.episode_data_index`
- Use `lerobot-dataset-viz` to visualize data

**Training Issues:**
- Verify observation/action spaces match between dataset and policy
- Check normalization statistics in checkpoint
- Monitor with `wandb.enable=true`

**Robot Control:**
- Use `lerobot-find-cameras` to debug camera connections
- Check motor IDs with `lerobot-setup-motors`
- Test with `lerobot-teleoperate` before recording

**Multi-Rate Control (SO-101 specific):**
- Recording at 30Hz, control at 180Hz, physics at 360Hz
- Jacobian-based XYZ control for first 3 joints
- Direct wrist control via keyboard (I/K and [/] keys)
