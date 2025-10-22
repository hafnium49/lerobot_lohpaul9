#!/bin/bash
# Run SO101 Residual RL Training with Zero Policy

echo "========================================="
echo "Starting SO101 Residual RL Training"
echo "Configuration: Zero Policy (Pure RL)"
echo "========================================="

# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export WANDB_API_KEY=$(grep WANDB_API_KEY .env | cut -d '=' -f2)
export MUJOCO_GL=egl  # For headless rendering
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0

# Run training with same configuration as previous run
python src/lerobot/scripts/train_so101_residual.py \
    --base-policy zero \
    --alpha 1.0 \
    --total-timesteps 100000 \
    --n-envs 4 \
    --output-dir runs/zero_policy_reproduced \
    --seed 42 \
    --learning-rate 0.0003 \
    --batch-size 64 \
    --n-steps 256 \
    --hidden-size 128 \
    --n-layers 2

echo "Training complete!"