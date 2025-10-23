#!/usr/bin/env python3
"""
Record dual-camera video of GR00T N1.5 controlling SO-101 in MuJoCo.

This script records the robot's behavior from two synchronized camera views:
- Top view (left): Bird's eye view of the workspace
- Wrist view (right): Robot's perspective from the wrist camera

Usage:
    python scripts/record_groot_dual_camera.py --output groot_dual_view.mp4
"""

import argparse
from pathlib import Path

import cv2
import mujoco as mj
import numpy as np

from lerobot.envs.so101_residual_env import SO101ResidualEnv
from lerobot.policies.groot_base_policy import GR00TBasePolicy


def record_dual_camera_video(
    model_path: str,
    output_path: str,
    n_episodes: int = 2,
    max_steps: int = 200,
    fps: int = 30,
    camera_names: tuple = ("top_view", "wrist_camera"),
    single_camera_size: tuple = (640, 480),
    seed: int = 42,
    separate_videos: bool = False,
) -> None:
    """
    Record dual-camera video of GR00T controlling SO-101.

    Args:
        model_path: Path to GR00T model
        output_path: Path to save video (or base path for separate videos)
        n_episodes: Number of episodes to record
        max_steps: Maximum steps per episode
        fps: Frames per second for video
        camera_names: Tuple of (left_camera, right_camera) names
        single_camera_size: Size of each camera view (width, height)
        seed: Random seed
        separate_videos: If True, save separate videos for each camera
    """

    print("=" * 80)
    print("Recording GR00T N1.5 Dual-Camera Control Video")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Max steps: {max_steps}")
    print(f"FPS: {fps}")
    print(f"Camera 1 (Left): {camera_names[0]}")
    print(f"Camera 2 (Right): {camera_names[1]}")
    print(f"Single view size: {single_camera_size[0]}x{single_camera_size[1]}")
    print(f"Mode: {'Separate videos' if separate_videos else 'Side-by-side'}")
    print()

    # Create environment
    env = SO101ResidualEnv(
        use_image_obs=True,
        image_size=(224, 224),
        camera_name_for_obs=camera_names[0],  # Use first camera for GR00T
        seed=seed,
    )
    print(f"✅ Environment created")

    # Create video renderers for both cameras
    cam1_id = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_CAMERA, camera_names[0])
    cam2_id = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_CAMERA, camera_names[1])

    renderer1 = mj.Renderer(env.model, single_camera_size[1], single_camera_size[0])
    renderer2 = mj.Renderer(env.model, single_camera_size[1], single_camera_size[0])
    print(f"✅ Video renderers created")

    # Load GR00T policy
    print(f"\nLoading GR00T policy...")
    policy = GR00TBasePolicy(
        model_path=model_path,
        device="cuda",
        expected_action_dim=6,
    )
    print(f"✅ GR00T policy loaded\n")

    # Initialize video writer(s)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    if separate_videos:
        # Create separate videos
        base_path = Path(output_path)
        output1 = str(base_path.parent / f"{base_path.stem}_{camera_names[0]}{base_path.suffix}")
        output2 = str(base_path.parent / f"{base_path.stem}_{camera_names[1]}{base_path.suffix}")

        video_writer1 = cv2.VideoWriter(output1, fourcc, fps, single_camera_size)
        video_writer2 = cv2.VideoWriter(output2, fourcc, fps, single_camera_size)

        print(f"Recording to:")
        print(f"  Camera 1: {output1}")
        print(f"  Camera 2: {output2}")
    else:
        # Create side-by-side video
        combined_width = single_camera_size[0] * 2
        combined_height = single_camera_size[1]
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))

        print(f"Recording to: {output_path}")
        print(f"Combined size: {combined_width}x{combined_height}")

    frame_count = 0

    print("\nRecording episodes...")
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

            # Render both cameras
            renderer1.update_scene(env.data, camera=cam1_id)
            renderer2.update_scene(env.data, camera=cam2_id)

            frame1_rgb = renderer1.render()
            frame2_rgb = renderer2.render()

            frame1_bgr = cv2.cvtColor(frame1_rgb, cv2.COLOR_RGB2BGR)
            frame2_bgr = cv2.cvtColor(frame2_rgb, cv2.COLOR_RGB2BGR)

            # Add text overlays
            text_y = 30
            overlay_info = [
                (f"Episode: {episode + 1}/{n_episodes}  Step: {step}/{max_steps}", (255, 255, 255)),
                (f"Return: {episode_return:.1f}", (255, 255, 255)),
                (f"GR00T N1.5", (255, 200, 0)),
            ]

            for frame, camera_name in [(frame1_bgr, camera_names[0]), (frame2_bgr, camera_names[1])]:
                y_pos = text_y
                for text, color in overlay_info:
                    cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_pos += 30

                # Add camera label
                cv2.putText(
                    frame,
                    camera_name.replace("_", " ").title(),
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1
                )

            # Write frame(s)
            if separate_videos:
                video_writer1.write(frame1_bgr)
                video_writer2.write(frame2_bgr)
            else:
                # Combine side-by-side
                combined_frame = np.hstack([frame1_bgr, frame2_bgr])
                video_writer.write(combined_frame)

            frame_count += 1

            if terminated or truncated:
                print(f"  Episode terminated at step {step}")
                break

        success = info.get("is_success", False)
        print(f"\n  Episode complete:")
        print(f"    Steps: {step + 1}")
        print(f"    Return: {episode_return:.2f}")
        print(f"    Success: {'✅ Yes' if success else '❌ No'}")

    # Finalize video(s)
    if separate_videos:
        video_writer1.release()
        video_writer2.release()
        print()
        print("=" * 80)
        print(f"✅ Videos saved:")
        print(f"   Camera 1 ({camera_names[0]}): {output1}")
        print(f"   Camera 2 ({camera_names[1]}): {output2}")
    else:
        video_writer.release()
        print()
        print("=" * 80)
        print(f"✅ Video saved: {output_path}")

    print(f"   Total frames: {frame_count}")
    print(f"   Duration: {frame_count / fps:.1f} seconds")
    print("=" * 80)

    renderer1.close()
    renderer2.close()
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Record dual-camera GR00T control video")
    parser.add_argument(
        "--model-path",
        type=str,
        default="phospho-app/gr00t-paper_return-7w9itxzsox",
        help="Path or HuggingFace repo ID for GR00T model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="groot_dual_view.mp4",
        help="Path to save video (or base path for separate videos)",
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
        "--camera1",
        type=str,
        default="top_view",
        help="First camera name (left side or separate video)",
    )
    parser.add_argument(
        "--camera2",
        type=str,
        default="wrist_camera",
        help="Second camera name (right side or separate video)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Width of each camera view",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Height of each camera view",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Save separate videos for each camera instead of side-by-side",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    record_dual_camera_video(
        model_path=args.model_path,
        output_path=args.output,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        fps=args.fps,
        camera_names=(args.camera1, args.camera2),
        single_camera_size=(args.width, args.height),
        seed=args.seed,
        separate_videos=args.separate,
    )


if __name__ == "__main__":
    main()
