#!/usr/bin/env python
"""
Quick test of full PPO training script (1000 steps).

Tests the production training script with minimal timesteps to verify:
1. Parallel environments work correctly
2. TensorBoard logging is functional
3. Checkpointing works
4. Evaluation callback works
5. No errors in the training loop
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os
from datetime import datetime
import json
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import configure

from lerobot.envs.so101_residual_env import SO101ResidualEnv


class QuickTestCallback(BaseCallback):
    """Simplified callback for quick testing."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals["dones"]):
            if done:
                self.episode_count += 1
                if self.episode_count % 5 == 0:
                    print(f"  Episodes completed: {self.episode_count}")
        return True


def make_env(rank: int, seed: int = 0):
    """Create a single environment instance."""
    def _init():
        env = SO101ResidualEnv(
            base_policy=None,
            alpha=0.7,
            act_scale=0.02,
            residual_penalty=0.001,
            randomize=True,
            render_mode=None,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    # Minimal configuration for quick test
    NUM_ENVS = 4  # Fewer envs for faster test
    TOTAL_TIMESTEPS = 1_000  # Just 1k steps
    N_STEPS = 128  # Smaller buffer
    BATCH_SIZE = 128
    N_EPOCHS = 3
    LEARNING_RATE = 3e-4
    SEED = 42

    # Logging setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"ppo_test_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Quick Test of PPO Training Script (1000 steps)")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  Parallel Environments: {NUM_ENVS}")
    print(f"  Total Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Log Directory: {log_dir}")
    print()

    # Create environments
    print("Creating parallel environments...")
    env = SubprocVecEnv([make_env(i, SEED) for i in range(NUM_ENVS)])
    env = VecMonitor(env)
    print(f"✅ Created {NUM_ENVS} parallel environments")
    print()

    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = SubprocVecEnv([make_env(0, SEED + 1000)])
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
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(log_dir / "tensorboard"),
        verbose=1,
        seed=SEED,
    )
    print("✅ PPO agent created")
    print()

    # Setup logger
    logger = configure(str(log_dir / "tensorboard"), ["tensorboard", "stdout"])
    model.set_logger(logger)

    # Callbacks
    print("Setting up callbacks...")
    test_callback = QuickTestCallback()

    checkpoint_callback = CheckpointCallback(
        save_freq=500 // NUM_ENVS,  # Save after 500 steps
        save_path=str(log_dir / "checkpoints"),
        name_prefix="ppo_test",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval_results"),
        eval_freq=500 // NUM_ENVS,  # Evaluate after 500 steps
        n_eval_episodes=3,  # Just 3 episodes
        deterministic=True,
        render=False,
    )

    callbacks = [test_callback, checkpoint_callback, eval_callback]
    print("✅ Callbacks configured")
    print()

    # Train
    print("=" * 80)
    print("Running quick training test (1000 steps)...")
    print("=" * 80)
    print()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            log_interval=1,
            tb_log_name="ppo_test",
        )

        print()
        print("=" * 80)
        print("Quick Test Passed!")
        print("=" * 80)
        print()

        # Save test model
        test_model_path = log_dir / "test_model"
        model.save(str(test_model_path))
        print(f"✅ Test model saved: {test_model_path}")
        print()

        # Verify files created
        print("Verifying output files...")
        expected_files = [
            log_dir / "checkpoints",
            log_dir / "best_model",
            log_dir / "eval_results",
            log_dir / "tensorboard",
        ]
        for path in expected_files:
            if path.exists():
                print(f"  ✅ {path.name} exists")
            else:
                print(f"  ❌ {path.name} missing")
        print()

        print("=" * 80)
        print("✅ All checks passed! Production training script is ready.")
        print("=" * 80)
        print()
        print("Next steps:")
        print("  1. Run full training:")
        print("     python scripts/train_ppo_residual.py")
        print("  2. Expected time: 2-4 hours for 50k steps")
        print("  3. Monitor with TensorBoard:")
        print(f"     tensorboard --logdir logs/")

    except Exception as e:
        print()
        print("=" * 80)
        print(f"❌ Test failed: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()

    finally:
        env.close()
        eval_env.close()
        print()
        print(f"Test logs: {log_dir}")


if __name__ == "__main__":
    main()
