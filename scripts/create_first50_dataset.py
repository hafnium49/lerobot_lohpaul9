#!/usr/bin/env python3
"""
Create a HuggingFace dataset with the first 50 episodes from paper_return dataset.
This represents the exact data used to train the GR00T N1.5 model.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, create_repo
from tqdm import tqdm

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_API_KEY")

if not HF_TOKEN:
    raise ValueError("HF_API_KEY not found in .env file")

# Configuration
SOURCE_DATASET = "Hafnium49/paper_return"
NEW_DATASET = "Hafnium49/paper_return_first50"
LOCAL_DIR = Path("./paper_return_first50")
NUM_EPISODES = 50

print("=" * 80)
print("Creating HuggingFace Dataset: First 50 Episodes")
print("=" * 80)
print(f"Source: {SOURCE_DATASET}")
print(f"Target: {NEW_DATASET}")
print(f"Episodes: 0-{NUM_EPISODES-1}")
print()

# Initialize HuggingFace API
api = HfApi(token=HF_TOKEN)

# Create local directory structure
LOCAL_DIR.mkdir(exist_ok=True)
(LOCAL_DIR / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
(LOCAL_DIR / "videos" / "chunk-000" / "observation.images.main").mkdir(parents=True, exist_ok=True)
(LOCAL_DIR / "meta").mkdir(parents=True, exist_ok=True)

print("✅ Local directory structure created")
print()

# Step 1: Download data files (parquet)
print("Step 1: Downloading data files (50 episodes)...")
data_files = []
for i in tqdm(range(NUM_EPISODES), desc="Data files"):
    filename = f"episode_{i:06d}.parquet"
    repo_path = f"data/chunk-000/{filename}"
    local_path = LOCAL_DIR / "data" / "chunk-000" / filename

    try:
        hf_hub_download(
            repo_id=SOURCE_DATASET,
            filename=repo_path,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )
        data_files.append(repo_path)
    except Exception as e:
        print(f"❌ Failed to download {repo_path}: {e}")

print(f"✅ Downloaded {len(data_files)} data files")
print()

# Step 2: Download video files (mp4)
print("Step 2: Downloading video files (50 episodes)...")
video_files = []
for i in tqdm(range(NUM_EPISODES), desc="Video files"):
    filename = f"episode_{i:06d}.mp4"
    repo_path = f"videos/chunk-000/observation.images.main/{filename}"
    local_path = LOCAL_DIR / "videos" / "chunk-000" / "observation.images.main" / filename

    try:
        hf_hub_download(
            repo_id=SOURCE_DATASET,
            filename=repo_path,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )
        video_files.append(repo_path)
    except Exception as e:
        print(f"❌ Failed to download {repo_path}: {e}")

print(f"✅ Downloaded {len(video_files)} video files")
print()

# Step 3: Download metadata files
print("Step 3: Downloading metadata files...")
meta_files = ["info.json", "stats.json", "episodes.jsonl", "tasks.jsonl"]
downloaded_meta = []
for filename in tqdm(meta_files, desc="Metadata"):
    repo_path = f"meta/{filename}"
    try:
        hf_hub_download(
            repo_id=SOURCE_DATASET,
            filename=repo_path,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )
        downloaded_meta.append(filename)
    except Exception as e:
        print(f"⚠️  Warning: Could not download {repo_path}: {e}")

print(f"✅ Downloaded {len(downloaded_meta)} metadata files")
print()

# Step 4: Create README
print("Step 4: Creating README.md...")
readme_content = f"""# Paper Return - First 50 Episodes (GR00T N1.5 Training Data)

## Overview

This dataset contains the **first 50 episodes** (episodes 0-49) from the [paper_return dataset](https://huggingface.co/datasets/{SOURCE_DATASET}).

This subset represents the **exact training data** used to fine-tune the [phospho-app/gr00t-paper_return-7w9itxzsox](https://huggingface.co/phospho-app/gr00t-paper_return-7w9itxzsox) GR00T N1.5 model.

## Dataset Contents

- **Episodes:** 50 (episodes 000000-000049)
- **Task:** Paper-in-square manipulation task using SO-101 robot
- **Data format:** LeRobot v2.1 dataset format
- **Total frames:** ~69,051 frames across 50 episodes

## Files Structure

```
paper_return_first50/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet  # Action and state data
│       ├── ...
│       └── episode_000049.parquet
├── videos/
│   └── chunk-000/
│       └── observation.images.main/
│           ├── episode_000000.mp4  # Camera observations
│           ├── ...
│           └── episode_000049.mp4
└── meta/
    ├── info.json           # Dataset metadata
    ├── stats.json          # Statistics for normalization
    ├── episodes.jsonl      # Episode information
    └── tasks.jsonl         # Task definitions
```

## Training Details

This dataset was used to train the GR00T N1.5 model with the following configuration:

- **Model:** GR00T N1.5 (3B parameters)
- **Training time:** ~3 hours on A100/H100 GPUs
- **Epochs:** 10
- **Batch size:** 49
- **Learning rate:** 0.0001
- **Platform:** phosphobot

## Related Resources

- **Full Dataset:** [{SOURCE_DATASET}](https://huggingface.co/datasets/{SOURCE_DATASET}) (206 episodes)
- **Trained Model:** [phospho-app/gr00t-paper_return-7w9itxzsox](https://huggingface.co/phospho-app/gr00t-paper_return-7w9itxzsox)
- **Model Documentation:** Training details and usage examples

## Usage

Load with LeRobot:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("{NEW_DATASET}")
print(f"Episodes: {{dataset.num_episodes}}")
print(f"Frames: {{len(dataset)}}")
```

## Citation

If you use this dataset, please cite:

- Original dataset: Hafnium49/paper_return
- GR00T model: phospho-app/gr00t-paper_return-7w9itxzsox
- GR00T foundation model: NVIDIA GR00T N1.5

## License

Same license as the original paper_return dataset.

## Dataset Creation

Created from episodes 0-49 of the paper_return dataset to represent the exact training data used for the GR00T N1.5 fine-tuning conducted on the phospho platform.
"""

readme_path = LOCAL_DIR / "README.md"
readme_path.write_text(readme_content)
print("✅ README.md created")
print()

# Step 5: Create repository and upload
print("Step 5: Creating HuggingFace repository and uploading...")
try:
    # Create repository
    create_repo(
        repo_id=NEW_DATASET,
        repo_type="dataset",
        token=HF_TOKEN,
        exist_ok=True,
        private=False
    )
    print(f"✅ Repository created: https://huggingface.co/datasets/{NEW_DATASET}")

    # Upload folder
    print("Uploading files to HuggingFace...")
    api.upload_folder(
        folder_path=str(LOCAL_DIR),
        repo_id=NEW_DATASET,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message=f"Add first {NUM_EPISODES} episodes (GR00T N1.5 training data)"
    )
    print("✅ Upload complete!")

except Exception as e:
    print(f"❌ Error creating/uploading repository: {e}")
    raise

print()
print("=" * 80)
print("✅ DATASET CREATION COMPLETE")
print("=" * 80)
print(f"Dataset URL: https://huggingface.co/datasets/{NEW_DATASET}")
print(f"Local copy: {LOCAL_DIR.absolute()}")
print()
print("Summary:")
print(f"  - Data files: {len(data_files)}")
print(f"  - Video files: {len(video_files)}")
print(f"  - Metadata files: {len(downloaded_meta)}")
print(f"  - Episodes: {NUM_EPISODES}")
print("=" * 80)
