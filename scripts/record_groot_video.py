#!/usr/bin/env python3
"""
Record video of GR00T N1.5 controlling SO-101 in MuJoCo.

This script records the robot's behavior as it's controlled by the
fine-tuned GR00T model and saves it as an MP4 video.

Usage:
    python scripts/record_groot_video.py --output groot_control.mp4
"""

import argparse
from pathlib import Path

import cv2
import mujoco as mj
import numpy as np

from lerobot.envs.so101_residual_env import SO101ResidualEnv
from lerobot.policies.groot_base_policy import GR00TBasePolicy


def record_groot_video(
    model_path: str,
    output_path: str,
    n_episodes: int = 2,
    max_steps: int = 200,
    fps: int = 30,
    camera_name: str = "top_view",
    video_size: tuple = (640, 480),
    seed: int = 42,
) -> None:
    """
    Record video of GR00T controlling SO-101.

    Args:
        model_path: Path to GR00T model
        output_path: Path to save video
        n_episodes: Number of episodes to record
        max_steps: Maximum steps per episode
        fps: Frames per second for video
        camera_name: Camera to use for recording
        video_size: Size of video (width, height)
        seed: Random seed
    """

    print("=" * 80)
    print("Recording GR00T N1.5 Control Video")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Max steps: {max_steps}")
    print(f"FPS: {fps}")
    print(f"Camera: {camera_name}")
    print(f"Video size: {video_size[0]}x{video_size[1]}")
    print()

    # Create environment
    env = SO101ResidualEnv(
        use_image_obs=True,
        image_size=(224, 224),
        camera_name_for_obs="top_view",
        seed=seed,
    )
    print(f"✅ Environment created")

    # Create video renderer
    cam_id = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_CAMERA, camera_name)
    video_renderer = mj.Renderer(env.model, video_size[1], video_size[0])
    print(f"✅ Video renderer created ({camera_name})")

    # Load GR00T policy
    print(f"\nLoading GR00T policy...")
    policy = GR00TBasePolicy(
        model_path=model_path,
        device="cuda",
        expected_action_dim=6,
    )
    print(f"✅ GR00T policy loaded\n")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, video_size)

    frame_count = 0

    print("Recording episodes...")
    print("=" * 80)

    for episode in range(n_episodes):
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        print("-" * 80)

        # Reset
        obs_dict, _ = env.reset()
        episode_return = 0.0

        for step in range(max_steps):
            # Get GR00T action
            image = obs_dict["image"]
            action = policy.predict(image)

            # Print progress
            if step % 50 == 0:
                print(f"  Step {step:3d}: Recording frame {frame_count}...")

            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(action)
            episode_return += reward

            # Render and save frame
            video_renderer.update_scene(env.data, camera=cam_id)
            frame_rgb = video_renderer.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Add text overlay with episode/step info
            text_y = 30
            cv2.putText(
                frame_bgr,
                f"Episode: {episode + 1}/{n_episodes}  Step: {step}/{max_steps}",
                (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame_bgr,
                f"Return: {episode_return:.1f}",
                (10, text_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame_bgr,
                f"GR00T N1.5 Base Policy",
                (10, text_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 200, 0),
                2,
            )

            # Write frame
            video_writer.write(frame_bgr)
            frame_count += 1

            if terminated or truncated:
                print(f"  Episode terminated at step {step}")
                break

        success = info.get("is_success", False)
        print(f"\n  Episode complete:")
        print(f"    Steps: {step + 1}")
        print(f"    Return: {episode_return:.2f}")
        print(f"    Success: {'✅ Yes' if success else '❌ No'}")

    # Finalize video
    video_writer.release()
    video_renderer.close()
    env.close()

    print()
    print("=" * 80)
    print(f"✅ Video saved: {output_path}")
    print(f"   Total frames: {frame_count}")
    print(f"   Duration: {frame_count / fps:.1f} seconds")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Record GR00T control video")
    parser.add_argument(
        "--model-path",
        type=str,
        default="phospho-app/gr00t-paper_return-7w9itxzsox",
        help="Path or HuggingFace repo ID for GR00T model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="groot_so101_control.mp4",
        help="Path to save video",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="Number of episodes to record",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="top_view",
        help="Camera name for recording",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Video width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    record_groot_video(
        model_path=args.model_path,
        output_path=args.output,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        fps=args.fps,
        camera_name=args.camera,
        video_size=(args.width, args.height),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
