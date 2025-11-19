#!/usr/bin/env python3
"""
Fix metadata for the paper_return_first50 dataset.
This script corrects the episode count mismatch in the metadata files.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, upload_file
from tqdm import tqdm

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_API_KEY")

if not HF_TOKEN:
    raise ValueError("HF_API_KEY not found in .env file")

# Configuration
DATASET_ID = "Hafnium49/paper_return_first50"
LOCAL_DIR = Path("./metadata_fix_temp")
NUM_EPISODES = 50

print("=" * 80)
print("Fixing Metadata for paper_return_first50 Dataset")
print("=" * 80)
print(f"Dataset: {DATASET_ID}")
print(f"Correcting to: {NUM_EPISODES} episodes")
print()

# Initialize HuggingFace API
api = HfApi(token=HF_TOKEN)

# Create local temp directory
LOCAL_DIR.mkdir(exist_ok=True)
(LOCAL_DIR / "meta").mkdir(parents=True, exist_ok=True)

print("Step 1: Downloading current metadata files...")
metadata_files = ["info.json", "episodes.jsonl"]
for filename in metadata_files:
    print(f"  Downloading {filename}...")
    hf_hub_download(
        repo_id=DATASET_ID,
        filename=f"meta/{filename}",
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        token=HF_TOKEN
    )
print("✅ Downloaded metadata files")
print()

# Step 2: Fix info.json
print("Step 2: Fixing info.json...")
info_path = LOCAL_DIR / "meta" / "info.json"
with open(info_path, 'r') as f:
    info_data = json.load(f)

print(f"  Current total_episodes: {info_data.get('total_episodes', 'N/A')}")
print(f"  Current splits: {info_data.get('splits', 'N/A')}")

# Calculate actual frame count from episodes.jsonl
episodes_path = LOCAL_DIR / "meta" / "episodes.jsonl"
total_frames = 0
with open(episodes_path, 'r') as f:
    for i, line in enumerate(f):
        if i >= NUM_EPISODES:
            break
        episode = json.loads(line.strip())
        # Get frame count from episode length
        if 'length' in episode:
            total_frames += episode['length']
        elif 'frame_end' in episode and 'frame_start' in episode:
            total_frames += episode['frame_end'] - episode['frame_start']

# Update info.json
info_data['total_episodes'] = NUM_EPISODES
info_data['splits'] = {"train": f"0:{NUM_EPISODES}"}
if total_frames > 0:
    info_data['total_frames'] = total_frames
    print(f"  Calculated total_frames: {total_frames}")

# Write updated info.json
with open(info_path, 'w') as f:
    json.dump(info_data, f, indent=2)

print(f"  ✅ Updated total_episodes to: {NUM_EPISODES}")
print(f"  ✅ Updated splits to: {info_data['splits']}")
print()

# Step 3: Fix episodes.jsonl (keep only first 50 episodes)
print("Step 3: Fixing episodes.jsonl...")
episodes_path = LOCAL_DIR / "meta" / "episodes.jsonl"
episodes_content = []

with open(episodes_path, 'r') as f:
    for i, line in enumerate(f):
        if i >= NUM_EPISODES:
            print(f"  Removed episodes {i} through end of file")
            break
        episodes_content.append(line.strip())

# Write filtered episodes.jsonl
with open(episodes_path, 'w') as f:
    for line in episodes_content:
        f.write(line + '\n')

print(f"  ✅ Kept first {len(episodes_content)} episodes")
print()

# Step 4: Upload fixed metadata files
print("Step 4: Uploading fixed metadata to HuggingFace...")

for filename in metadata_files:
    local_path = LOCAL_DIR / "meta" / filename
    remote_path = f"meta/{filename}"

    print(f"  Uploading {filename}...")
    try:
        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=DATASET_ID,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message=f"Fix metadata: Set episode count to {NUM_EPISODES}"
        )
        print(f"  ✅ Uploaded {filename}")
    except Exception as e:
        print(f"  ❌ Failed to upload {filename}: {e}")
        raise

print()
print("=" * 80)
print("✅ METADATA FIX COMPLETE")
print("=" * 80)
print(f"Dataset URL: https://huggingface.co/datasets/{DATASET_ID}")
print()
print("Summary of changes:")
print(f"  - Set total_episodes to: {NUM_EPISODES}")
print(f"  - Updated splits to: 0:{NUM_EPISODES}")
print(f"  - Filtered episodes.jsonl to first {NUM_EPISODES} episodes")
if total_frames > 0:
    print(f"  - Recalculated total_frames to: {total_frames}")
print()
print("The dataset should now work correctly with GR00T training!")
print("=" * 80)