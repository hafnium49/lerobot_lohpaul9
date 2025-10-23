#!/usr/bin/env python3
"""
Evaluate GR00T base policy transfer to SO-101 simulation.

This script tests the fine-tuned GR00T model on the paper return task
without any residual RL corrections (alpha=0). This measures the quality
of sim-to-real transfer and helps decide whether to proceed with residual
RL training or fall back to the Jacobian IK baseline.

Decision Criteria:
- success_rate < 5%:   Domain gap too large - DON'T use GR00T
- success_rate < 20%:  Marginal transfer - proceed with caution
- success_rate >= 20%: Good transfer - proceed with residual training

Usage:
    python scripts/eval_groot_base_only.py --n-episodes 100
"""

import argparse
import time
from pathlib import Path

import numpy as np

from lerobot.envs.so101_residual_env import SO101ResidualEnv
from lerobot.policies.groot_base_policy import GR00TBasePolicy


def evaluate_base_policy(
    model_path: str,
    n_episodes: int = 100,
    max_steps: int = 300,
    seed: int = 42,
    render_every: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Evaluate GR00T base policy on SO-101 paper return task.

    Args:
        model_path: Path to GR00T model
        n_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        seed: Random seed
        render_every: Save video every N episodes (0 to disable)
        verbose: Print progress

    Returns:
        results: Dictionary with evaluation metrics
    """

    print("=" * 80)
    print("GR00T Base Policy Evaluation")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Random seed: {seed}")
    print()

    # Create environment with image observations
    env = SO101ResidualEnv(
        use_image_obs=True,
        image_size=(224, 224),
        camera_name_for_obs="top_view",
        seed=seed,
    )
    print(f"✅ Environment created")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
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

    # Episode tracking
    episode_returns = []
    episode_lengths = []
    episode_successes = []
    final_distances = []
    min_distances = []

    # Time tracking
    start_time = time.time()
    inference_times = []

    print("Starting evaluation...")
    print("-" * 80)

    for episode in range(n_episodes):
        # Reset environment
        obs_dict = env.reset()
        episode_return = 0.0
        episode_length = 0
        min_distance = float("inf")

        for step in range(max_steps):
            # Extract image from observation dict
            image = obs_dict["image"]
            state = obs_dict["state"]

            # Get action from GR00T policy (alpha=0, so no residual)
            t0 = time.time()
            action = policy.predict(image)
            inference_time = time.time() - t0
            inference_times.append(inference_time)

            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            episode_length += 1

            # Track minimum distance to goal
            if "distance_to_goal" in info:
                min_distance = min(min_distance, info["distance_to_goal"])

            # Check termination
            if terminated or truncated:
                break

        # Episode complete - record metrics
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        min_distances.append(min_distance)

        # Check success (based on final info)
        success = info.get("is_success", False)
        episode_successes.append(success)
        final_distance = info.get("distance_to_goal", min_distance)
        final_distances.append(final_distance)

        # Print progress
        if verbose and (episode + 1) % 10 == 0:
            recent_success_rate = np.mean(episode_successes[-10:]) * 100
            recent_avg_return = np.mean(episode_returns[-10:])
            recent_avg_distance = np.mean(final_distances[-10:])
            avg_inference_time = np.mean(inference_times) * 1000

            print(
                f"Episode {episode + 1:3d}/{n_episodes}: "
                f"Success rate (last 10): {recent_success_rate:5.1f}% | "
                f"Avg return: {recent_avg_return:7.2f} | "
                f"Avg distance: {recent_avg_distance:.4f} | "
                f"Inference: {avg_inference_time:.1f}ms"
            )

    # Compute final statistics
    elapsed_time = time.time() - start_time

    print("-" * 80)
    print()
    print("=" * 80)
    print("Evaluation Results")
    print("=" * 80)

    success_rate = np.mean(episode_successes) * 100
    avg_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    avg_length = np.mean(episode_lengths)
    avg_final_distance = np.mean(final_distances)
    avg_min_distance = np.mean(min_distances)
    avg_inference = np.mean(inference_times) * 1000
    fps = sum(episode_lengths) / elapsed_time

    print(f"Success Rate:           {success_rate:6.2f}%")
    print(f"Average Return:         {avg_return:7.2f} ± {std_return:.2f}")
    print(f"Average Episode Length: {avg_length:7.2f} steps")
    print(f"Avg Final Distance:     {avg_final_distance:.4f} m")
    print(f"Avg Min Distance:       {avg_min_distance:.4f} m")
    print()
    print(f"Average Inference Time: {avg_inference:.1f} ms")
    print(f"Evaluation Speed:       {fps:.1f} steps/sec")
    print(f"Total Time:             {elapsed_time/60:.1f} minutes")
    print()

    # Decision criteria
    print("=" * 80)
    print("Transfer Quality Assessment")
    print("=" * 80)

    if success_rate < 5:
        print("❌ Domain gap too large - DON'T use GR00T")
        print("   Recommendation: Use Jacobian IK baseline instead")
        decision = "reject"
    elif success_rate < 20:
        print("⚠️  Marginal transfer - proceed with caution")
        print("   Recommendation: Consider using Jacobian IK baseline")
        print("   Or proceed with residual RL training with low expectations")
        decision = "marginal"
    else:
        print("✅ Good transfer - proceed with residual training")
        print("   Recommendation: Use GR00T as base policy for residual RL")
        decision = "accept"

    print()

    # Prepare results dictionary
    results = {
        "model_path": model_path,
        "n_episodes": n_episodes,
        "success_rate": success_rate,
        "avg_return": avg_return,
        "std_return": std_return,
        "avg_length": avg_length,
        "avg_final_distance": avg_final_distance,
        "avg_min_distance": avg_min_distance,
        "avg_inference_ms": avg_inference,
        "fps": fps,
        "elapsed_time": elapsed_time,
        "decision": decision,
        "episode_returns": episode_returns,
        "episode_successes": episode_successes,
        "final_distances": final_distances,
        "min_distances": min_distances,
    }

    # Clean up
    env.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate GR00T base policy transfer")
    parser.add_argument(
        "--model-path",
        type=str,
        default="phospho-app/gr00t-paper_return-7w9itxzsox",
        help="Path or HuggingFace repo ID for GR00T model",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON (optional)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity",
    )

    args = parser.parse_args()

    # Run evaluation
    results = evaluate_base_policy(
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        verbose=not args.quiet,
    )

    # Save results if requested
    if args.output:
        import json

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python types for JSON serialization
        json_results = {
            k: (v.tolist() if isinstance(v, np.ndarray) else float(v) if isinstance(v, np.floating) else v)
            for k, v in results.items()
        }

        with open(output_path, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to: {output_path}")
        print()

    return results


if __name__ == "__main__":
    main()
