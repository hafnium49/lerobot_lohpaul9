#!/usr/bin/env python3
"""
Analyze joint-4 delta issues in paper_return dataset.

This script investigates why the paper_return dataset causes joint-4 deltas
to exceed 1.5708 rad during inference, while paper_return_front_view works fine.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset

# Constants
DELTA_THRESHOLD = 1.5708  # π/2 rad - the safety limit
REPO_IDS = [
    "Hafnium49/paper_return",
    "Hafnium49/paper_return_front_view"
]

def normalize_angle(angle):
    """Normalize angle to [-π, π] range."""
    return np.arctan2(np.sin(angle), np.cos(angle))

def compute_delta_stats(positions):
    """Compute frame-to-frame deltas and statistics."""
    deltas = np.diff(positions, axis=0)

    # Also compute wrapped deltas (accounting for ±π boundary)
    wrapped_deltas = np.array([normalize_angle(d) for d in deltas])

    return {
        'raw_deltas': deltas,
        'wrapped_deltas': wrapped_deltas,
        'max_raw_delta': np.max(np.abs(deltas)),
        'max_wrapped_delta': np.max(np.abs(wrapped_deltas)),
        'mean_raw_delta': np.mean(np.abs(deltas)),
        'mean_wrapped_delta': np.mean(np.abs(wrapped_deltas)),
        'std_raw_delta': np.std(deltas),
        'std_wrapped_delta': np.std(wrapped_deltas),
        'num_exceeding_threshold': np.sum(np.abs(deltas) > DELTA_THRESHOLD),
        'num_wrapped_exceeding_threshold': np.sum(np.abs(wrapped_deltas) > DELTA_THRESHOLD),
    }

def analyze_dataset(repo_id):
    """Analyze a single dataset for joint position statistics."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {repo_id}")
    print(f"{'='*80}")

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset(repo_id, split="train")

    print(f"Total frames: {len(ds)}")

    # Extract state and action data
    states = np.array(ds['observation.state'])
    actions = np.array(ds['action'])
    episodes = np.array(ds['episode_index'])

    num_joints = states.shape[1]
    print(f"Number of joints: {num_joints}")

    # Global statistics
    results = {
        'repo_id': repo_id,
        'total_frames': len(ds),
        'num_joints': num_joints,
        'num_episodes': len(np.unique(episodes)),
        'joint_stats': {},
        'problematic_frames': [],
    }

    # Per-joint analysis
    for joint_idx in range(num_joints):
        print(f"\n--- Joint {joint_idx} ---")

        state_positions = states[:, joint_idx]
        action_positions = actions[:, joint_idx]

        # Basic statistics
        state_stats = {
            'min': float(np.min(state_positions)),
            'max': float(np.max(state_positions)),
            'mean': float(np.mean(state_positions)),
            'std': float(np.std(state_positions)),
            'range': float(np.max(state_positions) - np.min(state_positions)),
        }

        # Delta analysis
        state_delta_stats = compute_delta_stats(state_positions)
        action_delta_stats = compute_delta_stats(action_positions)

        results['joint_stats'][f'joint_{joint_idx}'] = {
            'state': state_stats,
            'state_deltas': {
                'max_raw': float(state_delta_stats['max_raw_delta']),
                'max_wrapped': float(state_delta_stats['max_wrapped_delta']),
                'mean_raw': float(state_delta_stats['mean_raw_delta']),
                'mean_wrapped': float(state_delta_stats['mean_wrapped_delta']),
                'std_raw': float(state_delta_stats['std_raw_delta']),
                'std_wrapped': float(state_delta_stats['std_wrapped_delta']),
                'exceeding_threshold_raw': int(state_delta_stats['num_exceeding_threshold']),
                'exceeding_threshold_wrapped': int(state_delta_stats['num_wrapped_exceeding_threshold']),
            },
            'action_deltas': {
                'max_raw': float(action_delta_stats['max_raw_delta']),
                'max_wrapped': float(action_delta_stats['max_wrapped_delta']),
                'mean_raw': float(action_delta_stats['mean_raw_delta']),
                'mean_wrapped': float(action_delta_stats['mean_wrapped_delta']),
                'std_raw': float(action_delta_stats['std_raw_delta']),
                'std_wrapped': float(action_delta_stats['std_wrapped_delta']),
                'exceeding_threshold_raw': int(action_delta_stats['num_exceeding_threshold']),
                'exceeding_threshold_wrapped': int(action_delta_stats['num_wrapped_exceeding_threshold']),
            }
        }

        print(f"  State range: [{state_stats['min']:.4f}, {state_stats['max']:.4f}] (Δ={state_stats['range']:.4f})")
        print(f"  State max delta (raw): {state_delta_stats['max_raw_delta']:.4f}")
        print(f"  State max delta (wrapped): {state_delta_stats['max_wrapped_delta']:.4f}")
        print(f"  Frames exceeding threshold (raw): {state_delta_stats['num_exceeding_threshold']}")
        print(f"  Frames exceeding threshold (wrapped): {state_delta_stats['num_wrapped_exceeding_threshold']}")

    # Special focus on joint-4
    print(f"\n{'*'*80}")
    print(f"JOINT-4 DETAILED ANALYSIS")
    print(f"{'*'*80}")

    joint4_states = states[:, 4]
    joint4_actions = actions[:, 4]

    # Find all frames with large deltas
    state_deltas = np.diff(joint4_states)
    problematic_indices = np.where(np.abs(state_deltas) > DELTA_THRESHOLD)[0]

    print(f"\nTotal problematic frames (delta > {DELTA_THRESHOLD:.4f}): {len(problematic_indices)}")

    if len(problematic_indices) > 0:
        print("\nFirst 20 problematic transitions:")
        print(f"{'Frame':<8} {'Episode':<10} {'From':<12} {'To':<12} {'Delta':<12} {'Wrapped Δ':<12}")
        print("-" * 80)

        for idx in problematic_indices[:20]:
            frame_from = idx
            frame_to = idx + 1
            from_val = joint4_states[frame_from]
            to_val = joint4_states[frame_to]
            delta = to_val - from_val
            wrapped_delta = normalize_angle(delta)
            episode = episodes[frame_to]

            print(f"{frame_to:<8} {episode:<10} {from_val:<12.4f} {to_val:<12.4f} {delta:<12.4f} {wrapped_delta:<12.4f}")

            results['problematic_frames'].append({
                'frame': int(frame_to),
                'episode': int(episode),
                'from_value': float(from_val),
                'to_value': float(to_val),
                'delta': float(delta),
                'wrapped_delta': float(wrapped_delta),
            })

    # Per-episode analysis for joint-4
    print(f"\nPer-episode joint-4 statistics:")
    print(f"{'Episode':<10} {'Frames':<10} {'Min':<12} {'Max':<12} {'Range':<12} {'Max Δ':<12} {'Issues':<10}")
    print("-" * 90)

    episode_stats = []
    for ep in np.unique(episodes):
        ep_mask = episodes == ep
        ep_positions = joint4_states[ep_mask]
        ep_deltas = np.diff(ep_positions)

        ep_stat = {
            'episode': int(ep),
            'num_frames': int(np.sum(ep_mask)),
            'min': float(np.min(ep_positions)),
            'max': float(np.max(ep_positions)),
            'range': float(np.max(ep_positions) - np.min(ep_positions)),
            'max_delta': float(np.max(np.abs(ep_deltas))) if len(ep_deltas) > 0 else 0.0,
            'num_issues': int(np.sum(np.abs(ep_deltas) > DELTA_THRESHOLD)) if len(ep_deltas) > 0 else 0,
        }
        episode_stats.append(ep_stat)

        if ep_stat['num_issues'] > 0 or ep_stat['max_delta'] > 1.0:  # Show episodes with issues or large deltas
            print(f"{ep:<10} {ep_stat['num_frames']:<10} {ep_stat['min']:<12.4f} {ep_stat['max']:<12.4f} "
                  f"{ep_stat['range']:<12.4f} {ep_stat['max_delta']:<12.4f} {ep_stat['num_issues']:<10}")

    results['joint4_episode_stats'] = episode_stats

    return results

def main():
    """Main analysis function."""
    print("Joint-4 Delta Analysis")
    print("=" * 80)
    print(f"Threshold: {DELTA_THRESHOLD:.4f} rad (π/2)")

    all_results = {}

    for repo_id in REPO_IDS:
        try:
            results = analyze_dataset(repo_id)
            all_results[repo_id] = results
        except Exception as e:
            print(f"\nERROR analyzing {repo_id}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_file = Path("joint4_analysis_results.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")

    # Generate comparison summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")

    for repo_id, results in all_results.items():
        print(f"\n{repo_id}:")
        print(f"  Total episodes: {results['num_episodes']}")
        print(f"  Total frames: {results['total_frames']}")

        j4_stats = results['joint_stats']['joint_4']
        print(f"  Joint-4 state range: [{j4_stats['state']['min']:.4f}, {j4_stats['state']['max']:.4f}]")
        print(f"  Joint-4 max raw delta: {j4_stats['state_deltas']['max_raw']:.4f}")
        print(f"  Joint-4 max wrapped delta: {j4_stats['state_deltas']['max_wrapped']:.4f}")
        print(f"  Frames exceeding threshold (raw): {j4_stats['state_deltas']['exceeding_threshold_raw']}")
        print(f"  Frames exceeding threshold (wrapped): {j4_stats['state_deltas']['exceeding_threshold_wrapped']}")
        print(f"  Total problematic frames: {len(results['problematic_frames'])}")

if __name__ == "__main__":
    main()
