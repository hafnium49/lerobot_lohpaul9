#!/usr/bin/env python3
"""
Analyze paper_return_calibrate dataset by reading all parquet files directly.
Compare with paper_return dataset.
"""

import json
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
import pyarrow.parquet as pq

def analyze_dataset_from_parquet(repo_id):
    """Analyze dataset by reading all parquet files directly."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {repo_id}")
    print(f"{'='*80}")

    # List all parquet files
    print("Finding parquet files...")
    files = list_repo_files(repo_id, repo_type="dataset")
    parquet_files = sorted([f for f in files if f.endswith('.parquet') and 'data/chunk' in f])
    print(f"Found {len(parquet_files)} episode files")

    # Collect all states and actions
    all_states = []
    all_actions = []

    print("Loading episodes...")
    for i, pfile in enumerate(parquet_files):
        if i % 10 == 0:
            print(f"  Loaded {i}/{len(parquet_files)} episodes...")

        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=pfile,
            repo_type='dataset'
        )

        table = pq.read_table(file_path)
        df = table.to_pandas()

        # Convert list columns to arrays
        states = np.vstack(df['observation.state'].values)
        actions = np.vstack(df['action'].values)

        all_states.append(states)
        all_actions.append(actions)

    # Concatenate all data
    all_states = np.vstack(all_states)
    all_actions = np.vstack(all_actions)

    print(f"  Loaded {len(parquet_files)}/{len(parquet_files)} episodes")
    print(f"\nTotal frames: {len(all_states)}")
    print(f"Number of joints: {all_states.shape[1]}\n")

    # Analyze each joint (SQL-style, 1-indexed)
    num_joints = all_states.shape[1]
    n_rows = len(all_states)
    results = []

    print(f"{'Joint':<8} {'Action Min':<15} {'Action Max':<15} {'State Min':<15} {'State Max':<15} "
          f"{'Rows':<10} {'>π/2':<10} {'>π':<10} {'%>π/2':<10}")
    print("-" * 120)

    for joint_id in range(1, num_joints + 1):
        idx = joint_id - 1  # Convert to 0-indexed

        action_col = all_actions[:, idx]
        state_col = all_states[:, idx]

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

        print(f"{joint_id:<8} {action_min:<15.6f} {action_max:<15.6f} {state_min:<15.6f} {state_max:<15.6f} "
              f"{n_rows:<10} {action_over_halfpi:<10} {action_over_pi:<10} {pct_action_over_halfpi:<10.4f}")

    return {
        "repo_id": repo_id,
        "total_frames": n_rows,
        "num_joints": num_joints,
        "joint_stats": results
    }

def load_previous_results():
    """Load previous analysis results from paper_return."""
    result_file = Path("dataset_comparison_results.json")
    if result_file.exists():
        with open(result_file, 'r') as f:
            data = json.load(f)
            # Find paper_return results
            for result in data:
                if 'paper_return' in result['repo_id'] and 'calibrate' not in result['repo_id']:
                    return result
    return None

def compare_joint5(old_result, new_result):
    """Compare joint-5 between datasets."""
    print(f"\n{'='*80}")
    print("COMPARISON: Joint-5 (Wrist Roll - The Problematic Joint)")
    print(f"{'='*80}\n")

    # Extract joint-5 stats
    old_j5 = next(j for j in old_result['joint_stats'] if j['joint_id'] == 5)
    new_j5 = next(j for j in new_result['joint_stats'] if j['joint_id'] == 5)

    print(f"{'Metric':<30} {'Old (paper_return)':<25} {'New (calibrate)':<25} {'Change':<15}")
    print("-" * 95)

    metrics = [
        ('State Min', 'state_min'),
        ('State Max', 'state_max'),
        ('State Range', lambda d: d['state_max'] - d['state_min']),
        ('Action Min', 'action_min'),
        ('Action Max', 'action_max'),
        ('Actions > π/2', 'action_over_halfpi'),
        ('% Actions > π/2', 'pct_action_over_halfpi'),
    ]

    for label, key in metrics:
        if callable(key):
            old_val = key(old_j5)
            new_val = key(new_j5)
        else:
            old_val = old_j5[key]
            new_val = new_j5[key]

        if isinstance(old_val, (int, float)):
            change = new_val - old_val
            if isinstance(old_val, int) and 'pct' not in str(key):
                print(f"{label:<30} {old_val:<25} {new_val:<25} {change:+.0f}")
            else:
                print(f"{label:<30} {old_val:<25.6f} {new_val:<25.6f} {change:+.6f}")
        else:
            print(f"{label:<30} {old_val:<25} {new_val:<25}")

    # Check inference compatibility
    problematic_state = 1.32
    old_includes = old_j5['state_min'] <= problematic_state <= old_j5['state_max']
    new_includes = new_j5['state_min'] <= problematic_state <= new_j5['state_max']

    print(f"\n{'*'*80}")
    print("Inference Compatibility Check")
    print(f"{'*'*80}")
    print(f"\nThe model was failing because joint-5 was at +1.32 rad during inference.")
    print(f"\nOld dataset (paper_return):")
    print(f"  Range: [{old_j5['state_min']:.4f}, {old_j5['state_max']:.4f}]")
    print(f"  Includes +1.32 rad: {'✓ YES' if old_includes else '✗ NO'}")
    print(f"\nNew dataset (paper_return_calibrate):")
    print(f"  Range: [{new_j5['state_min']:.4f}, {new_j5['state_max']:.4f}]")
    print(f"  Includes +1.32 rad: {'✓ YES' if new_includes else '✗ NO'}")

    if new_includes and not old_includes:
        print(f"\n✓✓✓ SUCCESS! Calibration FIXED the range coverage issue!")
        print(f"    The new dataset now includes the problematic +1.32 rad state.")
    elif not new_includes and not old_includes:
        print(f"\n✗✗✗ WARNING: +1.32 rad is still outside the training range.")
        print(f"    You may need to record more episodes or check calibration again.")
    elif new_includes and old_includes:
        print(f"\n✓ Both datasets include +1.32 rad (this shouldn't happen based on original error).")

    # Check reduction in large actions
    print(f"\n{'*'*80}")
    print("Action Distribution Improvement")
    print(f"{'*'*80}")
    reduction = old_j5['action_over_halfpi'] - new_j5['action_over_halfpi']
    pct_reduction = (reduction / old_j5['action_over_halfpi']) * 100 if old_j5['action_over_halfpi'] > 0 else 0

    print(f"\nActions with |value| > π/2:")
    print(f"  Old dataset: {old_j5['action_over_halfpi']:,} frames ({old_j5['pct_action_over_halfpi']:.2f}%)")
    print(f"  New dataset: {new_j5['action_over_halfpi']:,} frames ({new_j5['pct_action_over_halfpi']:.2f}%)")
    print(f"  Reduction: {reduction:,} frames ({pct_reduction:.1f}% improvement)")

def main():
    """Main analysis function."""
    print("Calibrated Dataset Analysis")
    print("=" * 80)

    # Load previous results
    old_result = load_previous_results()
    if not old_result:
        print("ERROR: Could not find previous paper_return analysis results.")
        print("Please run compare_datasets.py first on paper_return dataset.")
        return

    print(f"\nLoaded previous analysis for: {old_result['repo_id']}")

    # Analyze new calibrated dataset
    new_result = analyze_dataset_from_parquet("Hafnium49/paper_return_calibrate")

    # Save results
    output_file = Path("calibrated_dataset_analysis.json")
    with open(output_file, 'w') as f:
        json.dump({
            "old": old_result,
            "new": new_result
        }, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")

    # Compare joint-5
    compare_joint5(old_result, new_result)

if __name__ == "__main__":
    main()
