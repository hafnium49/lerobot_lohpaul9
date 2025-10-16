#!/usr/bin/env python
"""
Full PPO baseline training for SO-101 residual RL (Phase 5).

This script trains a residual RL policy from scratch using PPO with:
- 8 parallel environments for sample efficiency
- State-only observations (25D privileged information)
- Zero-action baseline (to be replaced with GR00T IL in Phase 3)
- Domain randomization (paper pose + friction)
- TensorBoard logging
- Periodic checkpoints and evaluation
- Target: 85-90% success rate on randomized starts

Expected training time: 2-4 hours for 50k steps (depending on hardware)
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import configure

from lerobot.envs.so101_residual_env import SO101ResidualEnv


class DetailedLoggingCallback(BaseCallback):
    """Callback for detailed logging of training metrics."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.residual_magnitudes = []

    def _on_step(self) -> bool:
        # Log residual action magnitudes
        if "actions" in self.locals:
            actions = self.locals["actions"]
            residual_mag = np.linalg.norm(actions, axis=-1).mean()
            self.residual_magnitudes.append(residual_mag)
            self.logger.record("train/residual_magnitude", residual_mag)

        # Check for done episodes
        for i, done in enumerate(self.locals["dones"]):
            if done:
                if len(self.locals["infos"][i]) > 0:
                    info = self.locals["infos"][i]

                    # Episode metrics
                    ep_reward = info.get("episode", {}).get("r", 0)
                    ep_length = info.get("episode", {}).get("l", 0)
                    ep_success = info.get("is_success", False)

                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    self.episode_successes.append(ep_success)

                    # Log to tensorboard
                    self.logger.record("rollout/ep_reward", ep_reward)
                    self.logger.record("rollout/ep_length", ep_length)
                    self.logger.record("rollout/ep_success", float(ep_success))

                    # Success rate over last 100 episodes
                    if len(self.episode_successes) >= 100:
                        recent_success_rate = np.mean(self.episode_successes[-100:])
                        self.logger.record("rollout/success_rate_100", recent_success_rate)

        return True

    def get_summary(self) -> dict:
        """Get summary statistics for this training run."""
        return {
            "total_episodes": len(self.episode_rewards),
            "mean_reward": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            "mean_length": float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0,
            "success_rate": float(np.mean(self.episode_successes)) if self.episode_successes else 0.0,
            "mean_residual_magnitude": float(np.mean(self.residual_magnitudes)) if self.residual_magnitudes else 0.0,
        }


def make_env(rank: int, seed: int = 0):
    """
    Create a single environment instance.

    Args:
        rank: Index of the environment in the parallel setup
        seed: Random seed for reproducibility
    """
    def _init():
        env = SO101ResidualEnv(
            base_policy=None,  # Zero-action baseline (Phase 3 will replace with GR00T)
            alpha=0.7,  # Residual blending factor
            act_scale=0.02,  # Action scaling (joint delta radians)
            residual_penalty=0.001,  # L2 penalty on residual actions
            randomize=True,  # Enable domain randomization
            render_mode=None,  # No rendering for speed
        )
        env.reset(seed=seed + rank)  # Unique seed per environment
        return env
    return _init


def main():
    # Configuration
    NUM_ENVS = 8  # Parallel environments for sample efficiency
    TOTAL_TIMESTEPS = 50_000  # Total training steps (can increase to 100k-500k)
    N_STEPS = 256  # Steps per environment before update
    BATCH_SIZE = 256  # Minibatch size
    N_EPOCHS = 10  # Optimization epochs per update
    LEARNING_RATE = 3e-4  # PPO learning rate
    SEED = 42  # Random seed for reproducibility

    # Logging setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"ppo_residual_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"PPO Residual RL Training - Phase 5")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  Parallel Environments: {NUM_ENVS}")
    print(f"  Total Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Steps per Update: {N_STEPS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs per Update: {N_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Random Seed: {SEED}")
    print(f"  Log Directory: {log_dir}")
    print()

    # Save configuration
    config = {
        "num_envs": NUM_ENVS,
        "total_timesteps": TOTAL_TIMESTEPS,
        "n_steps": N_STEPS,
        "batch_size": BATCH_SIZE,
        "n_epochs": N_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "timestamp": timestamp,
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
    }
    with open(log_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Creating parallel environments...")
    env = SubprocVecEnv([make_env(i, SEED) for i in range(NUM_ENVS)])
    env = VecMonitor(env)  # Wrap for episode statistics
    print(f"✅ Created {NUM_ENVS} parallel environments")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    print()

    # Create evaluation environment (single env, deterministic)
    print("Creating evaluation environment...")
    eval_env = SubprocVecEnv([make_env(0, SEED + 1000)])  # Different seed
    eval_env = VecMonitor(eval_env)
    print("✅ Evaluation environment created")
    print()

    # Create PPO agent
    print("Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE parameter
        clip_range=0.2,  # PPO clip range
        clip_range_vf=None,  # No value function clipping
        ent_coef=0.005,  # Entropy coefficient for exploration
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping
        use_sde=False,  # No state-dependent exploration
        sde_sample_freq=-1,
        target_kl=None,  # No KL constraint
        tensorboard_log=str(log_dir / "tensorboard"),
        verbose=1,
        seed=SEED,
    )
    print("✅ PPO agent created")
    print()

    # Setup TensorBoard logger
    logger = configure(str(log_dir / "tensorboard"), ["tensorboard", "stdout"])
    model.set_logger(logger)

    # Callbacks
    print("Setting up callbacks...")

    # 1. Detailed logging callback
    detailed_logger = DetailedLoggingCallback()

    # 2. Checkpoint callback (save every 10k steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000 // NUM_ENVS,  # Save every 10k timesteps
        save_path=str(log_dir / "checkpoints"),
        name_prefix="ppo_residual",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # 3. Evaluation callback (evaluate every 5k steps)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval_results"),
        eval_freq=5_000 // NUM_ENVS,  # Evaluate every 5k timesteps
        n_eval_episodes=10,  # 10 episodes per evaluation
        deterministic=True,
        render=False,
    )

    callbacks = [detailed_logger, checkpoint_callback, eval_callback]
    print("✅ Callbacks configured")
    print()

    # Train
    print("=" * 80)
    print("Starting training...")
    print(f"Expected updates: {TOTAL_TIMESTEPS // (NUM_ENVS * N_STEPS)}")
    print(f"Expected time: ~2-4 hours (hardware dependent)")
    print("=" * 80)
    print()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            log_interval=1,  # Log every update
            tb_log_name="ppo_residual",
        )

        print()
        print("=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print()

        # Save final model
        final_model_path = log_dir / "final_model"
        model.save(str(final_model_path))
        print(f"✅ Final model saved: {final_model_path}")

        # Save training summary
        summary = detailed_logger.get_summary()
        with open(log_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Training summary saved: {log_dir / 'training_summary.json'}")
        print()

        # Print summary
        print("Training Summary:")
        print(f"  Total Episodes: {summary['total_episodes']}")
        print(f"  Mean Reward: {summary['mean_reward']:.2f}")
        print(f"  Mean Episode Length: {summary['mean_length']:.1f}")
        print(f"  Success Rate: {summary['success_rate']*100:.1f}%")
        print(f"  Mean Residual Magnitude: {summary['mean_residual_magnitude']:.4f}")
        print()

        # Check if target achieved
        if summary['success_rate'] >= 0.85:
            print("🎉 Target achieved! Success rate >= 85%")
        else:
            print(f"⚠️  Target not reached. Success rate: {summary['success_rate']*100:.1f}% (target: 85%)")
            print("   Consider:")
            print("   - Increasing training timesteps (100k-500k)")
            print("   - Tuning hyperparameters (learning rate, entropy coefficient)")
            print("   - Adjusting domain randomization")
            print("   - Improving base policy (Phase 3: GR00T IL)")

    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("Training interrupted by user")
        print("=" * 80)
        print()

        # Save interrupted model
        interrupted_model_path = log_dir / "interrupted_model"
        model.save(str(interrupted_model_path))
        print(f"✅ Interrupted model saved: {interrupted_model_path}")

    finally:
        # Cleanup
        print()
        print("Closing environments...")
        env.close()
        eval_env.close()
        print("✅ Done!")
        print()
        print(f"Results saved in: {log_dir}")
        print()
        print("Next steps:")
        print("  1. Review TensorBoard logs:")
        print(f"     tensorboard --logdir {log_dir / 'tensorboard'}")
        print("  2. Evaluate best model:")
        print(f"     python scripts/eval_policy.py --model {log_dir / 'best_model' / 'best_model.zip'}")
        print("  3. If success rate < 85%, consider tuning hyperparameters or extending training")
        print("  4. Proceed to Phase 3 (GR00T IL stub) or Phase 6 (dataset integration)")


if __name__ == "__main__":
    main()
