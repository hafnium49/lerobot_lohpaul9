#!/usr/bin/env python
"""
SO101 Residual RL Training Script
Simplified version that handles GPU/CPU automatically
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Import our custom environment
from lerobot.envs.so101_residual_env import SO101ResidualEnv

def make_env(rank: int, seed: int = 42):
    """Create a single environment instance."""
    def _init():
        env = SO101ResidualEnv(
            base_policy=None,  # Zero-action baseline
            alpha=1.0,  # Full residual (no base policy)
            act_scale=0.02,  # Action scaling
            residual_penalty=0.001,  # L2 penalty on residual actions
            randomize=True,  # Enable domain randomization
            render_mode=None,  # No rendering for speed
        )
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    print("=" * 70)
    print("SO101 Residual RL Training - Zero Policy Baseline")
    print("=" * 70)

    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  GPU not available, using CPU")
        print("   Training will be slower but still functional")
    print()

    # Configuration
    config = {
        "base_policy": "zero",
        "alpha": 1.0,
        "total_timesteps": 100000,  # Reduced for testing
        "n_envs": 4,
        "learning_rate": 0.0003,
        "batch_size": 64,
        "n_steps": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.005,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "seed": 42,
    }

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"runs/zero_policy_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"  output_dir: {output_dir}")
    print(f"  device: {device}")
    print()

    # Initialize W&B
    print("Initializing Weights & Biases...")
    try:
        wandb.init(
            project="so101-residual-rl",
            name=f"zero_policy_{timestamp}",
            config=config,
            dir=str(output_dir),
        )
        print("✅ W&B initialized successfully")
    except Exception as e:
        print(f"⚠️  W&B initialization failed: {e}")
        print("   Continuing without W&B logging")
    print()

    # Create environments
    print(f"Creating {config['n_envs']} parallel environments...")
    env = SubprocVecEnv([make_env(i, config['seed']) for i in range(config['n_envs'])])
    env = VecMonitor(env, str(output_dir / "monitor"))

    # Create evaluation environment
    eval_env = SubprocVecEnv([make_env(0, config['seed'] + 1000)])
    eval_env = VecMonitor(eval_env, str(output_dir / "eval_monitor"))
    print(f"✅ Environments created")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    print()

    # Create PPO model
    print("Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        ent_coef=config['ent_coef'],
        vf_coef=config['vf_coef'],
        max_grad_norm=config['max_grad_norm'],
        tensorboard_log=str(output_dir / "tensorboard"),
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        verbose=1,
        seed=config['seed'],
        device=device,
    )
    print("✅ PPO agent created")
    print()

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // config['n_envs'],
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_residual",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=5000 // config['n_envs'],
        n_eval_episodes=10,
        deterministic=True,
    )

    callbacks = [checkpoint_callback, eval_callback]

    # Train
    print("=" * 70)
    print("Starting training...")
    print(f"Total timesteps: {config['total_timesteps']:,}")
    print(f"Expected time: ~10-20 minutes on {device.upper()}")
    print("=" * 70)
    print()

    try:
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=callbacks,
            log_interval=1,
            tb_log_name="ppo_residual",
        )

        print()
        print("=" * 70)
        print("✅ Training complete!")
        print("=" * 70)

        # Save final model
        model.save(str(output_dir / "final_model"))
        print(f"Model saved to: {output_dir / 'final_model.zip'}")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        model.save(str(output_dir / "interrupted_model"))
        print(f"Partial model saved to: {output_dir / 'interrupted_model.zip'}")

    finally:
        # Cleanup
        env.close()
        eval_env.close()
        if wandb.run:
            wandb.finish()
        print("\n✨ Done!")
        print(f"Results saved in: {output_dir}")
        print(f"View tensorboard: tensorboard --logdir {output_dir / 'tensorboard'}")

if __name__ == "__main__":
    main()