#!/usr/bin/env python3
"""
Compare paper_return and paper_return_calibrate datasets.
Mimics the SQL query to analyze joint ranges and statistics.
"""

import json
import numpy as np
from datasets import load_dataset
from pathlib import Path

REPO_IDS = [
    "Hafnium49/paper_return",
    "Hafnium49/paper_return_calibrate"
]

def analyze_joint_ranges(repo_id):
    """Analyze joint ranges similar to the SQL query."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {repo_id}")
    print(f"{'='*80}")

    # Load dataset - try LeRobot format first
    print("Loading dataset...")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset(repo_id)

        # Access the underlying HF dataset
        hf_ds = ds.hf_dataset
        print(f"Total frames: {len(hf_ds)}")

        # Extract state and action data
        states = np.array(hf_ds['observation.state'])
        actions = np.array(hf_ds['action'])
        n_rows = len(hf_ds)
    except Exception as e:
        print(f"LeRobot format failed, trying HuggingFace datasets: {e}")
        # Fallback to regular HF datasets
        ds = load_dataset(repo_id, split="train")
        print(f"Total frames: {len(ds)}")

        # Check if columns exist
        if 'observation.state' not in ds.column_names or 'action' not in ds.column_names:
            print(f"ERROR: Required columns not found. Available columns: {ds.column_names}")
            raise ValueError(f"Dataset {repo_id} does not have observation.state and action columns")

        # Extract state and action data
        states = np.array(ds['observation.state'])
        actions = np.array(ds['action'])
        n_rows = len(ds)

    num_joints = states.shape[1]

    print(f"Number of joints: {num_joints}\n")

    results = []

    # Mimic SQL query: joints 1-6 (1-indexed like SQL)
    for joint_id in range(1, num_joints + 1):
        # Python uses 0-indexing, so joint_id 1 = index 0
        idx = joint_id - 1

        action_col = actions[:, idx]
        state_col = states[:, idx]

        action_min = float(np.min(action_col))
        action_max = float(np.max(action_col))
        state_min = float(np.min(state_col))
        state_max = float(np.max(state_col))

        # Count values exceeding π/2 and π
        action_over_halfpi = int(np.sum(np.abs(action_col) > np.pi/2))
        action_over_pi = int(np.sum(np.abs(action_col) > np.pi))

        pct_action_over_halfpi = round(100.0 * action_over_halfpi / n_rows, 4)

        result = {
            "joint_id": joint_id,
            "action_min": action_min,
            "action_max": action_max,
            "state_min": state_min,
            "state_max": state_max,
            "n_rows": n_rows,
            "action_over_halfpi": action_over_halfpi,
            "action_over_pi": action_over_pi,
            "pct_action_over_halfpi": pct_action_over_halfpi
        }

        results.append(result)

        # Print in table format
        if joint_id == 1:
            print(f"{'Joint':<8} {'Action Min':<15} {'Action Max':<15} {'State Min':<15} {'State Max':<15} "
                  f"{'Rows':<10} {'>π/2':<10} {'>π':<10} {'%>π/2':<10}")
            print("-" * 120)

        print(f"{joint_id:<8} {action_min:<15.6f} {action_max:<15.6f} {state_min:<15.6f} {state_max:<15.6f} "
              f"{n_rows:<10} {action_over_halfpi:<10} {action_over_pi:<10} {pct_action_over_halfpi:<10.4f}")

    return {
        "repo_id": repo_id,
        "total_frames": n_rows,
        "num_joints": num_joints,
        "joint_stats": results
    }

def compare_datasets(results_list):
    """Compare joint-5 (joint_id=5, the problematic joint) between datasets."""
    print(f"\n{'='*80}")
    print("COMPARISON: Joint-5 (Wrist Roll - The Problematic Joint)")
    print(f"{'='*80}\n")

    if len(results_list) < 2:
        print("Need at least 2 datasets to compare")
        return

    # Extract joint-5 stats (joint_id=5, index 4 in Python)
    joint5_stats = []
    for result in results_list:
        for joint_stat in result['joint_stats']:
            if joint_stat['joint_id'] == 5:
                joint5_stats.append({
                    'dataset': result['repo_id'].split('/')[-1],
                    'repo_id': result['repo_id'],
                    **joint_stat
                })

    # Print comparison table
    print(f"{'Dataset':<30} {'State Min':<15} {'State Max':<15} {'State Range':<15} "
          f"{'Action >π/2':<15} {'% >π/2':<10}")
    print("-" * 105)

    for stat in joint5_stats:
        state_range = stat['state_max'] - stat['state_min']
        print(f"{stat['dataset']:<30} {stat['state_min']:<15.6f} {stat['state_max']:<15.6f} "
              f"{state_range:<15.6f} {stat['action_over_halfpi']:<15} {stat['pct_action_over_halfpi']:<10.4f}%")

    # Detailed comparison
    print(f"\n{'*'*80}")
    print("DETAILED COMPARISON")
    print(f"{'*'*80}\n")

    if len(joint5_stats) == 2:
        old = joint5_stats[0]
        new = joint5_stats[1]

        print(f"Dataset 1: {old['dataset']}")
        print(f"  State range: [{old['state_min']:.4f}, {old['state_max']:.4f}]")
        print(f"  Range span: {old['state_max'] - old['state_min']:.4f} rad")
        print(f"  Actions exceeding π/2: {old['action_over_halfpi']} ({old['pct_action_over_halfpi']:.2f}%)")
        print(f"  Mean absolute state: {(abs(old['state_min']) + abs(old['state_max'])) / 2:.4f} rad")

        print(f"\nDataset 2: {new['dataset']}")
        print(f"  State range: [{new['state_min']:.4f}, {new['state_max']:.4f}]")
        print(f"  Range span: {new['state_max'] - new['state_min']:.4f} rad")
        print(f"  Actions exceeding π/2: {new['action_over_halfpi']} ({new['pct_action_over_halfpi']:.2f}%)")
        print(f"  Mean absolute state: {(abs(new['state_min']) + abs(new['state_max'])) / 2:.4f} rad")

        print(f"\nChanges after calibration:")
        print(f"  State min shift: {new['state_min'] - old['state_min']:+.4f} rad")
        print(f"  State max shift: {new['state_max'] - old['state_max']:+.4f} rad")
        print(f"  Range change: {(new['state_max'] - new['state_min']) - (old['state_max'] - old['state_min']):+.4f} rad")
        print(f"  Reduction in >π/2 actions: {old['action_over_halfpi'] - new['action_over_halfpi']} frames")
        print(f"  Percentage point change: {new['pct_action_over_halfpi'] - old['pct_action_over_halfpi']:+.4f}%")

        # Check if new range includes +1.32 rad (the inference problem state)
        problematic_state = 1.32
        old_includes = old['state_min'] <= problematic_state <= old['state_max']
        new_includes = new['state_min'] <= problematic_state <= new['state_max']

        print(f"\nInference compatibility check (joint at +1.32 rad):")
        print(f"  Old dataset includes +1.32 rad: {'✓ YES' if old_includes else '✗ NO'}")
        print(f"  New dataset includes +1.32 rad: {'✓ YES' if new_includes else '✗ NO'}")

        if new_includes and not old_includes:
            print(f"  → ✓ Calibration FIXED the range coverage issue!")
        elif not new_includes and not old_includes:
            print(f"  → ✗ Still outside training range - may need more episodes")

def main():
    """Main analysis function."""
    print("Joint Range Analysis - Comparing Datasets")
    print("=" * 80)

    all_results = []

    for repo_id in REPO_IDS:
        try:
            results = analyze_joint_ranges(repo_id)
            all_results.append(results)
        except Exception as e:
            print(f"\nERROR analyzing {repo_id}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_file = Path("dataset_comparison_results.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")

    # Compare datasets
    if len(all_results) >= 2:
        compare_datasets(all_results)

if __name__ == "__main__":
    main()
