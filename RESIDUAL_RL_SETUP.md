# Residual RL Environment Setup with uv

This guide documents the setup process for the SO101 residual reinforcement learning environment using `uv`.

## ✅ Setup Complete!

The environment has been successfully configured with:
- Python 3.10.18 virtual environment
- MuJoCo 3.3.6
- Stable-Baselines3 2.7.0
- Gymnasium 1.2.1
- All necessary dependencies

## Quick Start

### 1. Activate the Environment

```bash
source .venv/bin/activate
```

### 2. Test the Installation

```bash
python -c "import mujoco, stable_baselines3, gymnasium; print('All modules loaded!')"
```

### 3. Run Training

```bash
cd src
python lerobot/scripts/train_so101_residual.py \
    --base-policy jacobian \
    --alpha 0.5 \
    --n-envs 4 \
    --total-timesteps 10000
```

### 4. Evaluate Policy

```bash
cd src
python lerobot/scripts/eval_so101_residual.py \
    --model-path ../runs/residual_rl/*/best_model/best_model.zip \
    --n-episodes 10
```

## What Was Done

### 1. Added Dependencies to pyproject.toml

Added a new optional dependency group for residual RL:

```toml
[project.optional-dependencies]
# Residual RL
residual-rl = [
    "stable-baselines3[extra]>=2.3.0",
    "mujoco>=3.1.0",
]
```

### 2. Created Virtual Environment with uv

```bash
# Created Python 3.10 environment (avoiding compatibility issues)
uv venv --python 3.10

# Activated environment
source .venv/bin/activate
```

### 3. Installed Dependencies

Due to some dependency conflicts with the full LeRobot stack, we installed the minimal required packages:

```bash
# Core RL dependencies
uv pip install stable-baselines3[extra] mujoco gymnasium matplotlib tqdm

# LeRobot without problematic dependencies
pip install -e . --no-deps

# Essential LeRobot dependencies
pip install datasets huggingface-hub einops opencv-python-headless jsonlines draccus==0.10.0
```

## Benefits of Using uv

1. **Speed**: 10-100x faster than pip for dependency resolution
2. **Reproducibility**: Can create lock files for exact environment reproduction
3. **Python Management**: Automatically downloads and manages Python versions
4. **Compatibility**: Works seamlessly with pyproject.toml
5. **Isolation**: Clean virtual environment prevents system package conflicts

## Troubleshooting

### Module Import Errors

If you get import errors, ensure you're in the `src` directory:

```bash
cd src
python -c "import lerobot.envs.so101_residual_env"
```

### Dependency Conflicts

The setup uses a minimal dependency set to avoid conflicts. If you need additional LeRobot features:

```bash
# Try installing specific extras
uv pip install gymnasium mujoco

# Or use pip for problematic packages
pip install <package> --no-deps
```

### GPU Support

The installation includes CUDA support. Verify with:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Next Steps

1. **Train a Policy**: Start with the Jacobian IK base policy
2. **Experiment**: Try different alpha values (0.3, 0.5, 0.7)
3. **Monitor**: Use TensorBoard to track training progress
4. **Evaluate**: Compare base vs residual performance

## Environment Details

- **Python**: 3.10.18 (managed by uv)
- **Virtual Environment**: `.venv/`
- **Key Packages**:
  - mujoco: 3.3.6
  - stable-baselines3: 2.7.0
  - gymnasium: 1.2.1
  - torch: 2.8.0 (with CUDA)

## Files Created

- `pyproject.toml`: Updated with residual-rl dependencies
- `.venv/`: Virtual environment with all packages
- `src/lerobot/envs/so101_residual_env.py`: Gymnasium environment
- `src/lerobot/envs/so101_base_policy.py`: Base policies
- `src/lerobot/scripts/train_so101_residual.py`: Training script
- `src/lerobot/scripts/eval_so101_residual.py`: Evaluation script

The environment is ready for residual RL experiments!