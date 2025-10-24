# Paper Return - First 50 Episodes (GR00T N1.5 Training Data)

## Overview

This dataset contains the **first 50 episodes** (episodes 0-49) from the [paper_return dataset](https://huggingface.co/datasets/Hafnium49/paper_return).

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

- **Full Dataset:** [Hafnium49/paper_return](https://huggingface.co/datasets/Hafnium49/paper_return) (206 episodes)
- **Trained Model:** [phospho-app/gr00t-paper_return-7w9itxzsox](https://huggingface.co/phospho-app/gr00t-paper_return-7w9itxzsox)
- **Model Documentation:** Training details and usage examples

## Usage

Load with LeRobot:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("Hafnium49/paper_return_first50")
print(f"Episodes: {dataset.num_episodes}")
print(f"Frames: {len(dataset)}")
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
