#!/bin/bash
# Quick-start script for baseline residual RL training

set -e  # Exit on error

echo "=============================================="
echo "SO-101 Residual RL - Baseline Training"
echo "=============================================="
echo ""

# Activate environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if in correct directory
if [ ! -d "src/lerobot" ]; then
    echo "❌ Error: Must run from repository root"
    exit 1
fi

# Install dependencies if needed
echo "Checking dependencies..."
python -c "import stable_baselines3" 2>/dev/null || {
    echo "Installing stable-baselines3..."
    uv pip install stable-baselines3[extra] gymnasium matplotlib tensorboard
}

echo "✅ Dependencies ready"
echo ""

# Create runs directory
mkdir -p runs

# Default parameters
BASE_POLICY=${1:-zero}
ALPHA=${2:-1.0}
TIMESTEPS=${3:-500000}
N_ENVS=${4:-4}
SEED=${5:-42}

echo "Training Configuration:"
echo "  Base Policy:    $BASE_POLICY"
echo "  Alpha:          $ALPHA"
echo "  Timesteps:      $TIMESTEPS"
echo "  Parallel Envs:  $N_ENVS"
echo "  Random Seed:    $SEED"
echo ""

OUTPUT_DIR="runs/${BASE_POLICY}_residual_$(date +%Y%m%d_%H%M%S)"

echo "Output Directory: $OUTPUT_DIR"
echo ""

# Launch TensorBoard in background
echo "Starting TensorBoard on http://localhost:6006"
tensorboard --logdir runs/ --port 6006 &
TENSORBOARD_PID=$!
echo "  (PID: $TENSORBOARD_PID)"
echo ""

echo "=============================================="
echo "Starting Training..."
echo "=============================================="
echo ""

# Run training
cd src
python lerobot/scripts/train_so101_residual.py \
    --base-policy $BASE_POLICY \
    --alpha $ALPHA \
    --total-timesteps $TIMESTEPS \
    --n-envs $N_ENVS \
    --seed $SEED \
    --output-dir ../$OUTPUT_DIR \
    --eval-freq 10000 \
    --save-freq 25000 \
    --n-eval-episodes 10

cd ..

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. View training curves: tensorboard --logdir runs/"
echo "  2. Evaluate policy: python src/lerobot/scripts/eval_so101_residual.py \\"
echo "                        --model-path $OUTPUT_DIR/best_model/best_model.zip"
echo ""

# Kill TensorBoard
if [ -n "$TENSORBOARD_PID" ]; then
    echo "Stopping TensorBoard (PID: $TENSORBOARD_PID)..."
    kill $TENSORBOARD_PID 2>/dev/null || true
fi

echo "✅ Done!"
