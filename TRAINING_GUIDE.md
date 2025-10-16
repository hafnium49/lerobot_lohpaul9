# PPO Residual RL Training Guide

**Date:** 2025-10-16
**Status:** Ready for Full Training Run

---

## Quick Start

### Option 1: Full Training (50k steps, ~2-4 hours)
```bash
source .venv/bin/activate
python scripts/train_ppo_residual.py
```

### Option 2: Quick Test (1k steps, ~5 seconds)
```bash
source .venv/bin/activate
python scripts/test_train_ppo_residual.py
```

### Option 3: Monitor Training
```bash
# In separate terminal
tensorboard --logdir logs/
# Then open http://localhost:6006
```

---

## Training Configuration

### Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Parallel Environments | 8 | SubprocVecEnv for sample efficiency |
| Total Timesteps | 50,000 | Can increase to 100k-500k |
| Steps per Update | 256 | Rollout buffer size per env |
| Batch Size | 256 | Minibatch size for optimization |
| Epochs | 10 | Optimization epochs per update |
| Learning Rate | 3e-4 | PPO learning rate |
| Gamma | 0.99 | Discount factor |
| GAE Lambda | 0.95 | Advantage estimation parameter |
| Clip Range | 0.2 | PPO clip range |
| Entropy Coefficient | 0.005 | Exploration bonus |
| Value Function Coef | 0.5 | Value loss weight |
| Max Grad Norm | 0.5 | Gradient clipping |

### Environment Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Base Policy | None (zero-action) | Will be replaced with GR00T IL |
| Alpha (residual blending) | 0.7 | Residual correction weight |
| Action Scale | 0.02 | Joint delta scale (radians) |
| Residual Penalty | 0.001 | L2 penalty on residual actions |
| Domain Randomization | Enabled | Paper pose + friction variation |
| Max Episode Steps | 400 | Episode timeout |

### Observation Space (25D State-Only)
```python
[
  joint_positions (6),      # qpos
  joint_velocities (6),     # qvel
  paper_pose (7),           # position + quaternion
  goal_vector (3),          # paper → tape center
  end_effector_pos (3),     # gripper position
]
```

### Action Space (6D Residual)
```python
residual_joint_deltas (6)  # Corrections to base policy
```

---

## Expected Training Behavior

### Phase 1: Random Exploration (0-5k steps)
- **Reward:** ~-200 to -150
- **Success Rate:** 0-5%
- **Behavior:** Robot explores workspace randomly

### Phase 2: Learning Basics (5k-20k steps)
- **Reward:** -150 to -50
- **Success Rate:** 5-30%
- **Behavior:** Robot starts approaching paper, occasional successes

### Phase 3: Refinement (20k-50k steps)
- **Reward:** -50 to +5
- **Success Rate:** 30-85%
- **Behavior:** Consistent paper manipulation, improving alignment

### Phase 4: Mastery (>50k steps, if extended)
- **Reward:** +5 to +10
- **Success Rate:** 85-95%
- **Behavior:** Reliable task completion

---

## Monitoring Metrics

### TensorBoard Metrics

**Primary Metrics:**
- `rollout/success_rate_100` - Success rate over last 100 episodes
- `rollout/ep_reward` - Episode reward
- `rollout/ep_length` - Episode length

**Training Metrics:**
- `train/approx_kl` - KL divergence (should stay < 0.1)
- `train/clip_fraction` - Fraction of clipped actions
- `train/entropy_loss` - Policy entropy (exploration)
- `train/policy_gradient_loss` - Policy loss
- `train/value_loss` - Value function loss

**Custom Metrics:**
- `train/residual_magnitude` - L2 norm of residual actions
- `eval/mean_reward` - Evaluation performance
- `eval/mean_ep_length` - Evaluation episode length

### Checkpoints

**Saved Automatically:**
- Every 10k steps: `logs/ppo_residual_TIMESTAMP/checkpoints/ppo_residual_10000_steps.zip`
- Best model: `logs/ppo_residual_TIMESTAMP/best_model/best_model.zip`
- Final model: `logs/ppo_residual_TIMESTAMP/final_model.zip`
- Interrupted model: `logs/ppo_residual_TIMESTAMP/interrupted_model.zip` (Ctrl+C)

### Log Files

**Generated Files:**
```
logs/ppo_residual_TIMESTAMP/
├── config.json              # Training configuration + git commit
├── training_summary.json    # Final statistics
├── checkpoints/             # Periodic model saves
├── best_model/              # Best performing model
├── eval_results/            # Evaluation logs
└── tensorboard/             # TensorBoard event files
```

---

## Troubleshooting

### Issue: Training is slow (< 100 FPS)
**Possible Causes:**
- Too many parallel environments for your CPU
- Rendering enabled (should be `render_mode=None`)

**Solutions:**
- Reduce `NUM_ENVS` from 8 to 4 or 2
- Verify rendering is disabled
- Close other applications

### Issue: Success rate not improving
**Possible Causes:**
- Insufficient training timesteps
- Suboptimal hyperparameters
- Task too difficult with zero-action baseline

**Solutions:**
- Increase `TOTAL_TIMESTEPS` to 100k-500k
- Reduce `LEARNING_RATE` to 1e-4
- Increase `ENTROPY_COEF` to 0.01 (more exploration)
- Implement Phase 3 (GR00T IL stub) for better base policy

### Issue: NaN/Inf in training
**Possible Causes:**
- Learning rate too high
- Gradient explosion

**Solutions:**
- Reduce `LEARNING_RATE` to 1e-4
- Reduce `MAX_GRAD_NORM` to 0.1
- Check environment for instability issues

### Issue: High residual magnitude
**Possible Causes:**
- Zero-action baseline is too weak
- Residual penalty too low

**Solutions:**
- Increase `residual_penalty` from 0.001 to 0.01
- Implement better base policy (Phase 3)
- Reduce `alpha` from 0.7 to 0.5

---

## Evaluation After Training

### Load and Evaluate Best Model
```python
from stable_baselines3 import PPO
from lerobot.envs.so101_residual_env import SO101ResidualEnv

# Load best model
model = PPO.load("logs/ppo_residual_TIMESTAMP/best_model/best_model.zip")

# Create environment
env = SO101ResidualEnv(
    base_policy=None,
    alpha=0.7,
    act_scale=0.02,
    residual_penalty=0.001,
    randomize=True,
    render_mode="human",  # Enable visualization
)

# Run evaluation
obs, _ = env.reset()
for _ in range(400):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"Episode finished. Success: {info.get('is_success', False)}")
        break
```

### Batch Evaluation (50 episodes)
```bash
# Create evaluation script (Phase 4)
python scripts/eval_policy.py --model logs/ppo_residual_TIMESTAMP/best_model/best_model.zip --episodes 50
```

---

## Next Steps After Training

### If Success Rate < 85%
1. **Extend Training:** Increase to 100k-500k steps
2. **Tune Hyperparameters:** Adjust learning rate, entropy, penalties
3. **Improve Base Policy:** Implement Phase 3 (GR00T IL stub)
4. **Enhance DR:** Implement Phase 2 (expand randomization)

### If Success Rate ≥ 85% ✅
1. **Save Results:** Commit `baseline_results.json`
2. **Proceed to Phase 3:** Implement GR00T IL stub
3. **Proceed to Phase 6:** Integrate HF dataset
4. **Proceed to Phase 8:** Add vision-based observations

---

## Hardware Requirements

### Minimum
- CPU: 4 cores (for 4 parallel envs)
- RAM: 8 GB
- GPU: Not required (MLP policy)
- Disk: 1 GB (logs + checkpoints)

### Recommended
- CPU: 8+ cores (for 8 parallel envs)
- RAM: 16 GB
- GPU: Optional (PPO with MLP doesn't benefit much)
- Disk: 5 GB (for extended runs)

### Expected Performance
- **4 envs:** ~100-150 FPS
- **8 envs:** ~150-200 FPS
- **16 envs:** ~200-250 FPS (diminishing returns)

---

## FAQ

**Q: Why state-only observations instead of vision?**
A: State-only (privileged information) is much faster to train and debug. Vision integration is Phase 8 after we establish a working baseline.

**Q: Why zero-action baseline instead of GR00T IL?**
A: Phase 1 focuses on physics and training pipeline. GR00T IL integration is Phase 3.

**Q: Can I use GPU?**
A: Yes, but MLP policies don't benefit much. You'll see a warning about poor GPU utilization (this is expected).

**Q: How do I resume interrupted training?**
A: Load the `interrupted_model.zip` and call `model.learn()` again with remaining timesteps.

**Q: What if training takes too long?**
A: Reduce `TOTAL_TIMESTEPS` to 10k-20k for a quick baseline, or reduce `NUM_ENVS` to speed up iteration.

---

## References

- [Stable-Baselines3 PPO Documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Implementation Progress](IMPLEMENTATION_PROGRESS.md)
- [SO-101 Residual Environment](src/lerobot/envs/so101_residual_env.py)
- [MuJoCo World](src/lerobot/envs/so101_assets/paper_square_realistic.xml)

---

**Last Updated:** 2025-10-16
**Author:** Claude Code
**Status:** Ready for Production Training 🚀
