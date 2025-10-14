#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Evaluation script for SO101 residual RL policies.

This script evaluates trained residual policies and compares them with base policies.
It can render videos, compute metrics, and generate comparison plots.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from tqdm import tqdm

from lerobot.envs.so101_base_policy import JacobianIKPolicy, ZeroPolicy
from lerobot.envs.so101_residual_env import SO101ResidualEnv


class PolicyEvaluator:
    """
    Evaluator for residual RL policies.
    """

    def __init__(
        self,
        env: SO101ResidualEnv,
        render: bool = False,
        save_video: bool = False,
        video_dir: Optional[Path] = None,
    ):
        """
        Initialize evaluator.

        Args:
            env: Environment to evaluate in
            render: Whether to render episodes
            save_video: Whether to save video frames
            video_dir: Directory to save videos
        """
        self.env = env
        self.render = render
        self.save_video = save_video
        self.video_dir = video_dir

        if save_video and video_dir:
            video_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_policy(
        self,
        policy,
        n_episodes: int = 100,
        deterministic: bool = True,
        progress_bar: bool = True,
    ) -> dict:
        """
        Evaluate a policy over multiple episodes.

        Args:
            policy: Policy to evaluate (PPO model or callable)
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic actions
            progress_bar: Show progress bar

        Returns:
            Dictionary with evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        residual_magnitudes = []
        final_distances = []
        time_to_success = []

        episodes = tqdm(range(n_episodes), desc="Evaluating", disable=not progress_bar)

        for episode_idx in episodes:
            obs, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_residuals = []
            frames = []

            done = False
            while not done:
                # Get action from policy
                if isinstance(policy, PPO):
                    action, _ = policy.predict(obs, deterministic=deterministic)
                elif callable(policy):
                    action = policy(obs)
                else:
                    raise ValueError(f"Unknown policy type: {type(policy)}")

                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Accumulate metrics
                episode_reward += reward
                episode_length += 1
                if "residual_action" in info:
                    episode_residuals.append(np.linalg.norm(info["residual_action"]))

                # Render if requested
                if self.render or self.save_video:
                    frame = self.env.render()
                    if self.save_video and frame is not None:
                        frames.append(frame)

            # Record episode metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_successes.append(info.get("success", False))
            final_distances.append(info.get("dist_to_goal", np.inf))

            if episode_residuals:
                residual_magnitudes.append(np.mean(episode_residuals))

            if info.get("success", False):
                time_to_success.append(episode_length)

            # Save video if requested
            if self.save_video and frames:
                self._save_video(frames, episode_idx, info.get("success", False))

            # Update progress bar
            episodes.set_postfix({
                "reward": f"{episode_reward:.1f}",
                "success": info.get("success", False),
            })

        # Compute statistics
        metrics = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "success_rate": np.mean(episode_successes),
            "mean_final_distance": np.mean(final_distances),
            "mean_residual_magnitude": np.mean(residual_magnitudes) if residual_magnitudes else 0,
            "mean_time_to_success": np.mean(time_to_success) if time_to_success else np.inf,
            "episode_rewards": episode_rewards,
            "episode_successes": episode_successes,
        }

        return metrics

    def compare_policies(
        self,
        policies: dict,
        n_episodes: int = 50,
        deterministic: bool = True,
    ) -> dict:
        """
        Compare multiple policies.

        Args:
            policies: Dictionary of {name: policy} to compare
            n_episodes: Number of episodes per policy
            deterministic: Use deterministic actions

        Returns:
            Dictionary with comparison results
        """
        results = {}

        for name, policy in policies.items():
            print(f"\nEvaluating {name}...")
            metrics = self.evaluate_policy(
                policy,
                n_episodes=n_episodes,
                deterministic=deterministic,
            )
            results[name] = metrics

            # Print summary
            print(f"  Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
            print(f"  Success rate: {metrics['success_rate']:.2%}")
            print(f"  Mean final distance: {metrics['mean_final_distance']:.3f}")

        return results

    def _save_video(self, frames: list, episode_idx: int, success: bool):
        """Save video frames to file."""
        if not self.video_dir:
            return

        try:
            import cv2

            # Create video writer
            height, width = frames[0].shape[:2]
            fps = 30
            success_str = "success" if success else "failure"
            video_path = self.video_dir / f"episode_{episode_idx:03d}_{success_str}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

            for frame in frames:
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(bgr_frame)

            writer.release()
            print(f"  Saved video to {video_path}")

        except ImportError:
            print("  Warning: OpenCV not installed, cannot save videos")


def plot_comparison(results: dict, save_path: Optional[Path] = None):
    """
    Plot comparison of policies.

    Args:
        results: Dictionary of evaluation results
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Metrics to plot
    metrics = [
        ("mean_reward", "Mean Reward"),
        ("success_rate", "Success Rate"),
        ("mean_final_distance", "Final Distance"),
        ("mean_length", "Episode Length"),
        ("mean_residual_magnitude", "Residual Magnitude"),
        ("mean_time_to_success", "Time to Success"),
    ]

    policy_names = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(policy_names)))

    for ax, (metric_key, metric_name) in zip(axes.flat, metrics):
        values = [results[name].get(metric_key, 0) for name in policy_names]

        if metric_key == "success_rate":
            values = [v * 100 for v in values]  # Convert to percentage

        bars = ax.bar(policy_names, values, color=colors)
        ax.set_title(metric_name)
        ax.set_ylabel(metric_name)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            format_str = f"{value:.1%}" if metric_key == "success_rate" else f"{value:.2f}"
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   format_str, ha='center', va='bottom')

    plt.suptitle("Policy Comparison: SO101 Paper-in-Square Task")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SO101 residual RL policies"
    )

    # Model arguments
    parser.add_argument("--model-path", type=str, required=True,
                      help="Path to trained model")
    parser.add_argument("--compare-base", action="store_true",
                      help="Compare with base policy alone")
    parser.add_argument("--compare-zero", action="store_true",
                      help="Compare with zero policy (pure RL)")

    # Environment arguments
    parser.add_argument("--xml-path", type=str, default=None,
                      help="Path to MuJoCo XML file")
    parser.add_argument("--alpha", type=float, default=0.5,
                      help="Residual blending factor")
    parser.add_argument("--act-scale", type=float, default=0.02,
                      help="Action scaling factor")
    parser.add_argument("--residual-penalty", type=float, default=0.001,
                      help="Residual penalty coefficient")
    parser.add_argument("--randomize", action="store_true",
                      help="Enable domain randomization")

    # Evaluation arguments
    parser.add_argument("--n-episodes", type=int, default=100,
                      help="Number of evaluation episodes")
    parser.add_argument("--deterministic", action="store_true",
                      help="Use deterministic actions")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")

    # Visualization arguments
    parser.add_argument("--render", action="store_true",
                      help="Render episodes")
    parser.add_argument("--save-video", action="store_true",
                      help="Save episode videos")
    parser.add_argument("--video-dir", type=str, default="videos",
                      help="Directory to save videos")
    parser.add_argument("--plot", action="store_true",
                      help="Generate comparison plots")
    parser.add_argument("--output-dir", type=str, default="eval_results",
                      help="Output directory for results")

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load trained model
    print(f"Loading model from {args.model_path}...")
    model = PPO.load(args.model_path)

    # Create base policy (for residual learning)
    base_policy = JacobianIKPolicy(max_delta=args.act_scale)

    # Create environment
    env = SO101ResidualEnv(
        xml_path=args.xml_path,
        base_policy=base_policy,
        alpha=args.alpha,
        act_scale=args.act_scale,
        residual_penalty=args.residual_penalty,
        randomize=args.randomize,
        seed=args.seed,
    )

    # Create evaluator
    video_dir = output_dir / args.video_dir if args.save_video else None
    evaluator = PolicyEvaluator(
        env,
        render=args.render,
        save_video=args.save_video,
        video_dir=video_dir,
    )

    # Prepare policies to evaluate
    policies = {"Residual RL": model}

    if args.compare_base:
        # Evaluate base policy alone (alpha=0)
        env_base = SO101ResidualEnv(
            xml_path=args.xml_path,
            base_policy=base_policy,
            alpha=0.0,  # No residual
            act_scale=args.act_scale,
            residual_penalty=0,
            randomize=args.randomize,
            seed=args.seed,
        )
        evaluator_base = PolicyEvaluator(env_base)
        zero_policy = ZeroPolicy()  # Dummy policy (base will handle everything)
        policies["Base Policy"] = zero_policy

    if args.compare_zero:
        # Evaluate pure RL without base (alpha=1, no base)
        env_zero = SO101ResidualEnv(
            xml_path=args.xml_path,
            base_policy=None,
            alpha=1.0,
            act_scale=args.act_scale,
            residual_penalty=0,
            randomize=args.randomize,
            seed=args.seed,
        )
        # Load model but use with no base
        policies["Pure RL"] = model

    # Run evaluation
    print(f"\nEvaluating {len(policies)} policies over {args.n_episodes} episodes each...")
    results = evaluator.compare_policies(
        policies,
        n_episodes=args.n_episodes,
        deterministic=args.deterministic,
    )

    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        # Convert numpy values to JSON-serializable format
        json_results = {}
        for name, metrics in results.items():
            json_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (list, np.ndarray)):
                    json_metrics[key] = [float(v) for v in value]
                else:
                    json_metrics[key] = float(value)
            json_results[name] = json_metrics
        json.dump(json_results, f, indent=2)
    print(f"\nSaved results to {results_file}")

    # Generate plots if requested
    if args.plot:
        plot_path = output_dir / "comparison_plot.png"
        plot_comparison(results, plot_path)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Success rate: {metrics['success_rate']:.2%}")
        print(f"  Mean final distance: {metrics['mean_final_distance']:.3f}")
        if metrics.get('mean_residual_magnitude', 0) > 0:
            print(f"  Mean residual magnitude: {metrics['mean_residual_magnitude']:.4f}")

    # Clean up
    env.close()


if __name__ == "__main__":
    main()