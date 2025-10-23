#!/usr/bin/env python3
"""
Visualize GR00T N1.5 controlling SO-101 in MuJoCo simulation.

This script runs the fine-tuned GR00T model and displays the robot
performing the paper return task with live rendering.

Usage:
    python scripts/visualize_groot_control.py --episodes 3 --render
"""

import argparse
import time
from pathlib import Path

import numpy as np

from lerobot.envs.so101_residual_env import SO101ResidualEnv
from lerobot.policies.groot_base_policy import GR00TBasePolicy


def visualize_groot_control(
    model_path: str,
    n_episodes: int = 3,
    max_steps: int = 300,
    render: bool = True,
    save_video: bool = False,
    video_path: str = "groot_control.mp4",
    seed: int = 42,
) -> None:
    """
    Visualize GR00T controlling SO-101.

    Args:
        model_path: Path to GR00T model
        n_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        render: Whether to render live (requires display)
        save_video: Whether to save video
        video_path: Path to save video
        seed: Random seed
    """

    print("=" * 80)
    print("GR00T N1.5 Base Policy Visualization")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Max steps: {max_steps}")
    print(f"Render: {render}")
    print(f"Save video: {save_video}")
    print()

    # Create environment with image observations and rendering
    render_mode = "human" if render else None
    env = SO101ResidualEnv(
        use_image_obs=True,
        image_size=(224, 224),
        camera_name_for_obs="top_view",
        seed=seed,
        render_mode=render_mode,
    )
    print(f"✅ Environment created")
    print(f"   Camera for observation: top_view")
    print(f"   Image size: 224×224")
    print()

    # Load GR00T base policy
    print(f"Loading GR00T base policy from {model_path}...")
    policy = GR00TBasePolicy(
        model_path=model_path,
        device="cuda",
        expected_action_dim=6,
    )
    print(f"✅ GR00T policy loaded")
    print()

    # Video recording setup
    video_frames = [] if save_video else None

    print("Starting visualization...")
    print("=" * 80)
    print()

    for episode in range(n_episodes):
        print(f"Episode {episode + 1}/{n_episodes}")
        print("-" * 80)

        # Reset environment
        obs_dict, reset_info = env.reset()
        episode_return = 0.0
        episode_length = 0
        min_distance = float("inf")

        # Episode loop
        for step in range(max_steps):
            # Extract image from observation dict
            image = obs_dict["image"]
            state = obs_dict["state"]

            # Get action from GR00T policy
            action = policy.predict(image)

            # Print action every 30 steps
            if step % 30 == 0:
                print(f"  Step {step:3d}: action = [{', '.join(f'{a:+.3f}' for a in action)}]")

            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            episode_length += 1

            # Track minimum distance
            if "distance_to_goal" in info:
                dist = info["distance_to_goal"]
                if not np.isinf(dist):
                    min_distance = min(min_distance, dist)

            # Render
            if render:
                env.render()
                time.sleep(0.01)  # Small delay for visibility

            # Save video frame if requested
            if save_video and video_frames is not None:
                frame = env.render()
                if frame is not None:
                    video_frames.append(frame)

            # Check termination
            if terminated or truncated:
                print(f"  Episode terminated at step {episode_length}")
                break

        # Episode summary
        success = info.get("is_success", False)
        final_distance = info.get("distance_to_goal", min_distance)

        print()
        print(f"  Episode Summary:")
        print(f"    Length:         {episode_length} steps")
        print(f"    Return:         {episode_return:.2f}")
        print(f"    Success:        {'✅ Yes' if success else '❌ No'}")
        print(f"    Min Distance:   {min_distance:.4f} m" if not np.isinf(min_distance) else "    Min Distance:   Not tracked")
        print(f"    Final Distance: {final_distance:.4f} m" if not np.isinf(final_distance) else "    Final Distance: Not tracked")
        print()

    # Save video if requested
    if save_video and video_frames:
        print(f"Saving video to {video_path}...")
        try:
            import cv2

            height, width = video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = 30
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            for frame in video_frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            out.release()
            print(f"✅ Video saved: {video_path}")
        except ImportError:
            print("⚠️  OpenCV not available, skipping video save")
        except Exception as e:
            print(f"⚠️  Failed to save video: {e}")

    print("=" * 80)
    print("Visualization complete!")
    print("=" * 80)

    # Clean up
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize GR00T controlling SO-101")
    parser.add_argument(
        "--model-path",
        type=str,
        default="phospho-app/gr00t-paper_return-7w9itxzsox",
        help="Path or HuggingFace repo ID for GR00T model",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to visualize",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable live rendering (requires display)",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save video to file",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="groot_control.mp4",
        help="Path to save video",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    visualize_groot_control(
        model_path=args.model_path,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        save_video=args.save_video,
        video_path=args.video_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
