#!/usr/bin/env python
"""
Record video of SO101 residual RL policy executing paper-in-square task.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from stable_baselines3 import PPO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.envs.so101_base_policy import JacobianIKPolicy
from lerobot.envs.so101_residual_env import SO101ResidualEnv


def record_episode_video(
    model_path: str,
    output_path: str = "policy_video.mp4",
    n_episodes: int = 3,
    deterministic: bool = True,
    alpha: float = 0.5,
    act_scale: float = 0.02,
    seed: int = 42,
    fps: int = 30,
):
    """
    Record video of trained policy executing episodes.

    Args:
        model_path: Path to trained PPO model
        output_path: Path to save video file
        n_episodes: Number of episodes to record
        deterministic: Use deterministic actions
        alpha: Residual blending factor
        act_scale: Action scaling
        seed: Random seed
        fps: Frames per second for video
    """
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # Create base policy
    base_policy = JacobianIKPolicy(max_delta=act_scale)

    # Create environment with RGB array rendering
    print("Creating environment...")
    env = SO101ResidualEnv(
        base_policy=base_policy,
        alpha=alpha,
        act_scale=act_scale,
        residual_penalty=0.0,
        randomize=True,
        render_mode="rgb_array",
        camera_name="top_view",
        seed=seed,
    )

    # Initialize video writer
    first_frame = env.render()
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\nRecording {n_episodes} episodes to {output_path}...")
    print(f"Video resolution: {width}x{height} @ {fps}fps\n")

    total_frames = 0

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

            # Render and save frame
            frame = env.render()
            if frame is not None:
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_frame)
                total_frames += 1

        # Print episode results
        success = info.get("success", False)
        dist_to_goal = info.get("dist_to_goal", 0)

        status_emoji = "✅" if success else "❌"
        print(f"  {status_emoji} Reward: {episode_reward:.2f} | Length: {episode_length} steps | Distance: {dist_to_goal:.3f}m")

    video_writer.release()
    env.close()

    duration = total_frames / fps
    print(f"\n✅ Video saved to: {output_path}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {duration:.1f} seconds")


def main():
    parser = argparse.ArgumentParser(
        description="Record video of SO101 residual RL policy"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="runs/jacobian_residual_wandb/so101_residual_jacobian_20251021_143356/best_model/best_model.zip",
        help="Path to trained model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="policy_execution.mp4",
        help="Output video file path",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=3,
        help="Number of episodes to record",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frames per second",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    record_episode_video(
        model_path=args.model_path,
        output_path=args.output,
        n_episodes=args.n_episodes,
        fps=args.fps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
