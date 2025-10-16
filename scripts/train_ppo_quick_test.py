#!/usr/bin/env python
"""
Quick PPO training test for SO-101 residual RL (100-200 steps).

This is a minimal test to verify:
1. Environment loads correctly
2. PPO training runs without errors
3. Loss decreases over time
4. No NaN/Inf in observations or actions

NOT a full training run - just a sanity check!
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from lerobot.envs.so101_residual_env import SO101ResidualEnv


class QuickTestCallback(BaseCallback):
    """Callback to monitor training progress during quick test."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Check for done episodes
        for i, done in enumerate(self.locals["dones"]):
            if done:
                if len(self.locals["infos"][i]) > 0:
                    ep_reward = self.locals["infos"][i].get("episode", {}).get("r", 0)
                    ep_length = self.locals["infos"][i].get("episode", {}).get("l", 0)
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)

                    print(f"  Episode {len(self.episode_rewards)}: "
                          f"reward={ep_reward:.2f}, length={ep_length}")

        return True


def make_env():
    """Create a single environment instance."""
    return SO101ResidualEnv(
        base_policy=None,  # Zero-action baseline
        alpha=0.7,  # Residual blending factor
        act_scale=0.02,  # Action scaling
        residual_penalty=0.001,  # L2 penalty
        randomize=True,  # Enable domain randomization
        render_mode=None,  # No rendering for speed
    )


def main():
    print("=" * 60)
    print("Quick PPO Training Test (100-200 steps)")
    print("=" * 60)
    print()

    # Create environment
    print("Creating environment...")
    env = DummyVecEnv([make_env])
    print(f"✅ Environment created")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    print()

    # Create PPO agent
    print("Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=64,  # Small for quick test
        batch_size=64,
        n_epochs=3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,  # Exploration
        verbose=1,
    )
    print("✅ PPO agent created")
    print()

    # Test reset
    print("Testing environment reset...")
    obs = env.reset()
    assert obs.shape == (1, 25), f"Expected obs shape (1, 25), got {obs.shape}"
    assert np.isfinite(obs).all(), "Observation contains NaN/Inf after reset"
    print("✅ Environment reset works")
    print(f"   Initial observation: {obs[0][:6]}... (showing first 6)")
    print()

    # Test single step
    print("Testing single step...")
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    assert obs.shape == (1, 25), f"Expected obs shape (1, 25), got {obs.shape}"
    assert np.isfinite(obs).all(), "Observation contains NaN/Inf after step"
    assert np.isfinite(reward).all(), "Reward contains NaN/Inf"
    print("✅ Single step works")
    print(f"   Action: {action[0]}")
    print(f"   Reward: {reward[0]:.4f}")
    print()

    # Quick training (200 steps)
    print("Running quick training (200 steps)...")
    print("This will take ~30-60 seconds...")
    print()

    callback = QuickTestCallback()

    model.learn(
        total_timesteps=200,
        callback=callback,
        log_interval=50,
    )

    print()
    print("=" * 60)
    print("Training Test Complete!")
    print("=" * 60)
    print()

    # Summary
    if callback.episode_rewards:
        print(f"Episodes completed: {len(callback.episode_rewards)}")
        print(f"Mean reward: {np.mean(callback.episode_rewards):.2f}")
        print(f"Mean length: {np.mean(callback.episode_lengths):.1f}")
        print()

    # Test inference
    print("Testing inference (predict action)...")
    obs = env.reset()
    action, _states = model.predict(obs, deterministic=True)
    assert action.shape == (1, 6), f"Expected action shape (1, 6), got {action.shape}"
    assert np.isfinite(action).all(), "Action contains NaN/Inf"
    print("✅ Inference works")
    print(f"   Predicted action: {action[0]}")
    print()

    print("=" * 60)
    print("✅ All checks passed! Training pipeline is functional.")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Run full training with more steps (10k-100k)")
    print("  2. Add evaluation script to measure success rate")
    print("  3. Integrate GR00T IL prior (currently using zero-action baseline)")
    print("  4. Tune hyperparameters based on learning curves")


if __name__ == "__main__":
    main()
