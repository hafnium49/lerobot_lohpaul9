#!/usr/bin/env python
"""
Quick visualization script for SO101 residual RL policy.

Shows the trained policy executing the paper-in-square task in real-time MuJoCo viewer.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.envs.so101_base_policy import JacobianIKPolicy
from lerobot.envs.so101_residual_env import SO101ResidualEnv


def visualize_policy(
    model_path: str,
    n_episodes: int = 5,
    deterministic: bool = True,
    alpha: float = 0.5,
    act_scale: float = 0.02,
    seed: int = 42,
):
    """
    Visualize trained policy in MuJoCo viewer.

    Args:
        model_path: Path to trained PPO model
        n_episodes: Number of episodes to run
        deterministic: Use deterministic actions
        alpha: Residual blending factor
        act_scale: Action scaling
        seed: Random seed
    """
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # Create base policy
    base_policy = JacobianIKPolicy(max_delta=act_scale)

    # Create environment with human rendering
    print("Creating environment with MuJoCo viewer...")
    env = SO101ResidualEnv(
        base_policy=base_policy,
        alpha=alpha,
        act_scale=act_scale,
        residual_penalty=0.0,  # No penalty during visualization
        randomize=True,
        render_mode="human",
        camera_name="top",
        seed=seed,
    )

    print(f"\nRunning {n_episodes} episodes with live visualization...")
    print("Close the MuJoCo window to stop.\n")

    for episode_idx in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0

        print(f"Episode {episode_idx + 1}/{n_episodes}")

        done = False
        while not done:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=deterministic)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Accumulate metrics
            episode_reward += reward
            episode_length += 1

            # Render (opens MuJoCo viewer window)
            env.render()

        # Print episode results
        success = info.get("success", False)
        dist_to_goal = info.get("dist_to_goal", 0)

        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length} steps")
        print(f"  Success: {success}")
        print(f"  Final distance to goal: {dist_to_goal:.3f}m")
        print()

    env.close()
    print("Visualization complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SO101 residual RL policy in MuJoCo viewer"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="runs/jacobian_residual_wandb/so101_residual_jacobian_20251021_143356/best_model/best_model.zip",
        help="Path to trained model",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=5,
        help="Number of episodes to visualize",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Residual blending factor (0=base only, 1=full residual)",
    )
    parser.add_argument(
        "--act-scale",
        type=float,
        default=0.02,
        help="Action scaling factor",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    visualize_policy(
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        deterministic=args.deterministic,
        alpha=args.alpha,
        act_scale=args.act_scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
