# GR00T N1.5 Base Policy Training Plan

**Date:** 2025-10-22
**Status:** BLOCKED - Requires Infrastructure Setup
**Estimated Setup Time:** 4-8 hours
**Estimated Training Time:** 2-3 hours (after setup)

---

## Executive Summary

This document outlines the plan for training a residual RL agent using the fine-tuned **GR00T N1.5 model** (`phospho-app/gr00t-paper_return-7w9itxzsox`) as the base policy. After thorough investigation, we identified **4 critical blockers** that must be resolved before training can proceed.

**Current Recommendation:** Complete and evaluate Jacobian IK baseline training first (currently running), then decide if GR00T setup investment (4-8 hours) is justified.

---

## Table of Contents

1. [Critical Blockers](#critical-blockers)
2. [Technical Requirements](#technical-requirements)
3. [Implementation Options](#implementation-options)
4. [Comparison: Jacobian vs GR00T](#comparison-jacobian-vs-groot)
5. [Recommended Path Forward](#recommended-path-forward)
6. [Setup Instructions](#setup-instructions)
7. [Training Configuration](#training-configuration)
8. [Validation Protocol](#validation-protocol)
9. [Expected Outcomes](#expected-outcomes)
10. [Risk Assessment](#risk-assessment)

---

## Critical Blockers

### Blocker #1: Model Loading Infrastructure ❌

**Problem:**
GR00T N1.5 is a custom architecture (`model_type: "gr00t_n1_5"`) that **cannot** be loaded with standard HuggingFace `transformers` library.

**Error When Attempting Load:**
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("phospho-app/gr00t-paper_return-7w9itxzsox")
# ❌ ValueError: The checkpoint you are trying to load has model type `gr00t_n1_5`
# but Transformers does not recognize this architecture.
```

**Current State:**
- ✅ Wrapper code exists: [groot_base_policy.py](../src/lerobot/policies/groot_base_policy.py)
- ✅ Test scripts ready: [test_groot_inference.py](../scripts/test_groot_inference.py)
- ❌ Model loading will fail without proper infrastructure

**Resolution Required:** Install either Isaac-GR00T package or set up Phosphobot inference server (see [Implementation Options](#implementation-options)).

---

### Blocker #2: Environment Observation Mismatch ❌

**Problem:**
GR00T requires **RGB images** as input, but the current `SO101ResidualEnv` uses **state-only observations** (25D privileged information).

**Current Observation Space:**
```python
obs = np.concatenate([
    joint_pos,       # (6,) - Joint positions
    joint_vel,       # (6,) - Joint velocities
    paper_pose,      # (7,) - Paper position + quaternion
    goal_vector,     # (3,) - Paper center to tape center
    ee_pos,          # (3,) - End-effector position
])  # Total: 25D, NO IMAGES
```

**GR00T Requirements:**
```python
# Input: RGB image
image = np.ndarray  # Shape: (H, W, 3), dtype: uint8
                   # Typical: (480, 640, 3) or (224, 224, 3)
                   # Values: [0, 255]

# From: Top-view or wrist-mounted camera
# Processing: GR00T's vision backbone (preprocessed by processor)
```

**Resolution Required:**
1. Modify `SO101ResidualEnv` to render camera images each step
2. Add image observation space to environment
3. Handle image preprocessing (resizing, normalization)
4. Create wrapper that provides images to GR00T and state to residual RL policy

**Impact:**
- Training speed: **10-20× slower** due to image rendering overhead
- Memory usage: **~100× higher** per observation
- Implementation time: **2-3 hours**

---

### Blocker #3: Sim-to-Real Domain Gap ⚠️

**Risk:**
GR00T was trained on **real SO-101 camera images** (physical robot, real lighting, real textures), but will receive **MuJoCo-rendered images** during simulation training.

**Domain Gap Factors:**
| Aspect | Real Robot (Training) | MuJoCo Sim (Inference) | Gap Severity |
|--------|----------------------|------------------------|--------------|
| **Lighting** | Natural/lab lighting, shadows | Uniform/artificial | High |
| **Textures** | Real materials, wear | Simplified meshes | Medium |
| **Colors** | Accurate RGB | Approximated | Medium |
| **Noise** | Camera noise, blur | Perfect rendering | Low |
| **Dynamics** | Real physics | Simulated physics | Medium |

**Expected Base Policy Performance:**
- **Best case**: 20-50% success (partial transfer)
- **Likely case**: 5-20% success (poor transfer)
- **Worst case**: 0-5% success (no transfer, random actions)

**Comparison:**
- Jacobian IK: **30-50% base success** (no domain gap)
- GR00T: **0-50% base success** (depends on transfer)

**Resolution Required:**
1. Run 100 evaluation episodes with GR00T base policy only (α=0)
2. Measure success rate to validate transfer quality
3. Only proceed with training if success rate >20%
4. If <20%: Fall back to Jacobian IK base policy

**Mitigation Strategies:**
- Add domain randomization (lighting, textures, colors)
- Use sim-to-real transfer techniques (domain adaptation)
- Fine-tune GR00T on MuJoCo-rendered images
- **Recommendation:** Use for real robot deployment instead of sim training

---

### Blocker #4: Integration Not Complete ❌

**Current Implementation Status:**

✅ **Completed Components:**
- `GR00TBasePolicy` wrapper class
- Modality-based action extraction
- Action dimension handling (6D or 7D)
- Absolute → relative conversion
- Gripper polarity inversion
- Test scripts and documentation

❌ **Missing Components:**
- Model loading mechanism (requires Isaac-GR00T or Phosphobot)
- Image observation support in `SO101ResidualEnv`
- Image-to-GR00T data flow
- GR00T-to-residual action blending with images
- Validation pipeline for sim-to-real transfer

**Resolution Required:** Complete missing components (estimated 4-6 hours).

---

## Technical Requirements

### Hardware Requirements

**Minimum for GR00T Training:**
- **CPU:** 8 cores (for parallel environments)
- **RAM:** 32 GB (16 GB for model + envs, 16 GB for image buffers)
- **GPU:** NVIDIA GPU with 12+ GB VRAM (for GR00T inference)
- **Disk:** 10 GB (model checkpoints + image buffers)

**Comparison with State-Only Training:**
| Resource | State-Only (Jacobian) | Image-Based (GR00T) | Ratio |
|----------|----------------------|---------------------|-------|
| **RAM** | 8 GB | 32 GB | 4× |
| **VRAM** | 2 GB (optional) | 12 GB (required) | 6× |
| **Disk** | 2 GB | 10 GB | 5× |
| **Training Speed** | ~700 FPS | ~30-70 FPS | 10-20× slower |

### Software Requirements

**Core Dependencies:**
```bash
# Already installed
- Python 3.10+
- PyTorch 2.2+ with CUDA
- MuJoCo 3.0+
- Stable-Baselines3
- LeRobot

# Required for GR00T (choose ONE)
# Option A: Isaac-GR00T
- NVIDIA Isaac-GR00T package

# Option B: Phosphobot
- phosphobot
- Running inference server
```

### Model Requirements

**GR00T N1.5 Model:**
- **HuggingFace:** `phospho-app/gr00t-paper_return-7w9itxzsox`
- **Size:** ~2-4 GB
- **Architecture:** Custom `gr00t_n1_5` (not in standard transformers)
- **Training Data:** Real SO-101 robot, paper manipulation task
- **Dataset:** `Hafnium49/paper_return` (6 demonstrations)

**Action Output Format:**
```python
outputs = {
    "action.single_arm": torch.Tensor,  # Shape: (batch, 16, 6)
    "action.gripper": torch.Tensor,     # Shape: (batch, 16, 1)
}

# For SO-101 (6 DOF total):
action = np.concatenate([
    outputs["action.single_arm"][0, 0, :5],  # First 5 arm joints
    outputs["action.gripper"][0, 0, :1]      # Gripper
])  # Result: (6,) numpy array
```

---

## Implementation Options

### Option A: Install Isaac-GR00T Package (Recommended)

**Pros:**
- ✅ Official NVIDIA implementation
- ✅ Full model support
- ✅ Better performance
- ✅ Local inference (no network dependency)

**Cons:**
- ❌ Requires setup time (2-3 hours)
- ❌ May have CUDA/dependency conflicts

**Installation Steps:**
```bash
# 1. Clone Isaac-GR00T repository
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T

# 2. Install package
pip install -e .

# 3. Test model loading
cd /path/to/lerobot_lohpaul9
python scripts/test_groot_inference.py

# Expected output:
# ✅ Model loaded successfully
# ✅ Inference successful
# Action shape: (6,)
# Action values: [...]
```

**Time Estimate:** 2-3 hours (clone + install + resolve conflicts + test)

---

### Option B: Use Phosphobot Inference Server

**Pros:**
- ✅ No local model setup
- ✅ Client-server architecture
- ✅ Easier dependency management

**Cons:**
- ❌ Requires running server
- ❌ Network latency overhead
- ❌ More complex architecture

**Installation Steps:**
```bash
# 1. Install Phosphobot
pip install phosphobot

# 2. Start inference server (in separate terminal)
python -m phosphobot.am.gr00t.serve \
    --model-path phospho-app/gr00t-paper_return-7w9itxzsox \
    --host 0.0.0.0 \
    --port 8000

# 3. Modify GR00TBasePolicy to use client API
# (requires code changes in groot_base_policy.py)

# 4. Test connection
curl http://localhost:8000/health
```

**Time Estimate:** 3-4 hours (install + server setup + client integration + test)

---

### Option C: Wait for HuggingFace Transformers Support

**Pros:**
- ✅ Standard API (AutoModel.from_pretrained)
- ✅ No extra dependencies

**Cons:**
- ❌ Unknown timeline (requires NVIDIA/HF cooperation)
- ❌ May never happen
- ❌ Cannot proceed with training

**Time Estimate:** Unknown / Indefinite

---

## Comparison: Jacobian vs GR00T

### Performance Comparison

| Metric | Jacobian IK | GR00T N1.5 | Winner |
|--------|-------------|------------|--------|
| **Setup Time** | 0 min (ready) | 4-8 hours | ✅ Jacobian |
| **Observation Type** | State (25D) | Images (H×W×3) | ✅ Jacobian |
| **Base Policy Success** | 30-50% | 0-50% (unknown) | ✅ Jacobian |
| **Training Speed** | ~700 FPS | ~30-70 FPS | ✅ Jacobian |
| **Training Time** | 23 min | 2-3 hours | ✅ Jacobian |
| **Memory Usage** | 8 GB | 32 GB | ✅ Jacobian |
| **Sim-to-Real** | No gap | Large gap | ✅ Jacobian |
| **Integration** | Complete | Blocked | ✅ Jacobian |
| **Task Knowledge** | Kinematics only | Learned from demos | ✅ GR00T |
| **Real Robot Use** | Generic IK | Task-specific | ✅ GR00T |
| **Expected Final Success** | 70-85% | 70-85% | 🟰 Tie |

### Cost-Benefit Analysis

**Jacobian IK:**
- **Investment:** 0 hours (already done)
- **Benefit:** 70-85% success in 23 min
- **ROI:** Immediate

**GR00T N1.5:**
- **Investment:** 4-8 hours setup + 2-3 hours training = 6-11 hours
- **Benefit:** 70-85% success (if transfer works)
- **ROI:** Only if you need task-specific knowledge OR planning real robot deployment

---

## Recommended Path Forward

### 🎯 Path 1: Wait for Jacobian Results (RECOMMENDED)

**Rationale:**
1. ✅ Jacobian training is **currently running** (shell ID: 9941c2)
2. ✅ Will complete in **~10 more minutes**
3. ✅ Expected: **70-85% success** rate
4. ✅ **Zero blockers** - works out of the box
5. ✅ No domain gap issues
6. ⏰ Already invested time - see results first

**Action Plan:**
```bash
# 1. Wait for Jacobian training to complete (~10 min)
# Check status:
# BashOutput tool on shell 9941c2

# 2. Analyze results
# - If ≥70% success: Mission accomplished ✅
# - If <70% success: Extend training or tune hyperparameters

# 3. Generate visualization
python record_policy_video.py \
    --model-path runs/jacobian_residual_improved_reward/.../best_model.zip \
    --output jacobian_residual_visualization.mp4 \
    --n-episodes 5

# 4. THEN decide on GR00T:
#    - If Jacobian ≥85%: Skip GR00T (good enough)
#    - If Jacobian 70-84%: Consider GR00T for marginal improvement
#    - If Jacobian <70%: GR00T unlikely to help (domain gap)
```

**Timeline:**
- Wait: 10 min
- Analysis: 5 min
- Visualization: 3 min
- **Total: 18 minutes**

---

### 🔧 Path 2: Invest in GR00T Setup

**When to Choose:**
- You need task-specific knowledge from demonstrations
- Planning real robot deployment (GR00T works better on real robot)
- Interested in vision-based policies for future work
- Have 6-11 hours available for setup + training

**Action Plan:**
```bash
# Phase 1: Infrastructure Setup (4-6 hours)
# 1. Install Isaac-GR00T
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T && pip install -e .

# 2. Test model loading
python scripts/test_groot_inference.py

# Phase 2: Environment Modification (2-3 hours)
# 1. Add image rendering to SO101ResidualEnv
# 2. Create image observation wrapper
# 3. Integrate GR00T with environment
# 4. Test end-to-end data flow

# Phase 3: Validation (1 hour)
# 1. Run 100 episodes with GR00T base only (α=0)
python scripts/eval_groot_base.py --n-episodes 100 --alpha 0.0

# 2. Check success rate:
#    - <5%: STOP - Domain gap too large
#    - 5-20%: CAUTION - May not help
#    - >20%: PROCEED - Good transfer

# Phase 4: Training (2-3 hours)
python src/lerobot/scripts/train_so101_residual.py \
    --base-policy il \
    --il-checkpoint phospho-app/gr00t-paper_return-7w9itxzsox \
    --alpha 0.5 \
    --total-timesteps 500000 \
    --n-envs 4 \
    --output-dir runs/groot_residual_improved_reward
```

**Timeline:**
- Setup: 4-6 hours
- Validation: 1 hour
- Training: 2-3 hours
- **Total: 7-10 hours**

**Risk:** May not work due to domain gap (see Blocker #3).

---

### 📊 Path 3: Parallel Approach

**Action Plan:**
1. ✅ Let Jacobian training finish (10 min)
2. 🔧 Start GR00T setup in parallel (4-6 hours)
3. 📊 Compare both approaches when complete

**Benefits:**
- Get immediate baseline (Jacobian)
- Prepare infrastructure for future (GR00T)
- Make informed decision with both results

**Timeline:**
- Jacobian: 10 min (already running)
- GR00T: 4-8 hours (parallel work)
- **Total: ~8 hours** (but Jacobian done first)

---

## Setup Instructions

### Step 1: Install Isaac-GR00T (Option A)

```bash
# Navigate to workspace
cd ~/

# Clone repository
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T

# Check requirements
cat requirements.txt

# Install package (editable mode)
pip install -e .

# Verify installation
python -c "import isaac_groot; print('✅ Isaac-GR00T installed')"
```

**Troubleshooting:**
```bash
# If CUDA version mismatch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# If dependency conflicts:
pip install --upgrade transformers diffusers

# If import errors:
export PYTHONPATH=$PYTHONPATH:$PWD
```

---

### Step 2: Test Model Loading

```bash
cd /path/to/lerobot_lohpaul9

# Run test script
python scripts/test_groot_inference.py

# Expected output:
# Loading GR00T model from phospho-app/gr00t-paper_return-7w9itxzsox
# Device: cuda
# ✅ GR00T model loaded successfully
# Testing inference on dummy image...
# ✅ Inference successful
# Action shape: (6,)
# Action values: [...]
# Action range: [-X.XXX, X.XXX]
```

**If Successful:**
- ✅ Model loading works
- ✅ Action extraction works
- ✅ Ready for environment integration

**If Failed:**
- ❌ Check error message
- ❌ Verify Isaac-GR00T installation
- ❌ Check CUDA/PyTorch compatibility
- ❌ Consider Phosphobot alternative (Option B)

---

### Step 3: Modify Environment for Image Observations

**Current Environment:** `src/lerobot/envs/so101_residual_env.py`

**Required Changes:**

```python
class SO101ResidualEnv(gym.Env):
    def __init__(
        self,
        # ... existing args ...
        use_image_obs: bool = False,  # NEW
        image_size: tuple = (224, 224),  # NEW
        camera_name: str = "top_view",  # NEW
    ):
        # ... existing init ...

        # NEW: Image observation configuration
        self.use_image_obs = use_image_obs
        self.image_size = image_size
        self.camera_name_for_obs = camera_name

        if use_image_obs:
            # Image observation space
            self.observation_space = Box(
                low=0,
                high=255,
                shape=(*image_size, 3),
                dtype=np.uint8
            )
        else:
            # State observation space (existing)
            self.observation_space = Box(...)

    def _get_obs(self) -> np.ndarray:
        if self.use_image_obs:
            # Render camera image
            image = self._render_camera(self.camera_name_for_obs)
            # Resize if needed
            if image.shape[:2] != self.image_size:
                image = cv2.resize(image, self.image_size)
            return image
        else:
            # Return state observation (existing)
            return self._get_state_obs()

    def _render_camera(self, camera_name: str) -> np.ndarray:
        # Get camera ID
        cam_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, camera_name)

        # Render
        renderer = mj.Renderer(self.model, self.image_size[0], self.image_size[1])
        renderer.update_scene(self.data, camera=cam_id)
        image = renderer.render()

        return image
```

**Estimated Time:** 2-3 hours (implement + test + debug)

---

### Step 4: Create GR00T Environment Wrapper

**File:** `src/lerobot/envs/so101_groot_wrapper.py`

```python
"""
Environment wrapper for using GR00T as base policy with residual RL.
"""

import numpy as np
import gymnasium as gym
from lerobot.policies.groot_base_policy import GR00TBasePolicy


class GR00TResidualWrapper(gym.Wrapper):
    """
    Wrapper that provides images to GR00T base policy and state to residual.
    """

    def __init__(
        self,
        env: gym.Env,
        groot_model_path: str,
        alpha: float = 0.5,
        device: str = "auto",
    ):
        super().__init__(env)

        # Load GR00T base policy
        self.groot_policy = GR00TBasePolicy(
            model_path=groot_model_path,
            device=device,
            expected_action_dim=6,
        )

        self.alpha = alpha

        # Observation space: state for residual RL
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(25,),
            dtype=np.float32
        )

    def step(self, residual_action: np.ndarray):
        # Get current image from environment
        image = self.env._render_camera("top_view")

        # Get base action from GR00T
        base_action = self.groot_policy.predict(image)

        # Blend actions
        total_action = base_action + self.alpha * residual_action

        # Step environment with total action
        obs, reward, terminated, truncated, info = self.env.step(total_action)

        # Return state observation for residual RL
        state_obs = self.env._get_state_obs()

        return state_obs, reward, terminated, truncated, info
```

---

### Step 5: Validate Sim-to-Real Transfer

**Script:** `scripts/eval_groot_base.py` (create new file)

```python
#!/usr/bin/env python
"""
Evaluate GR00T base policy alone to validate sim-to-real transfer.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.envs.so101_residual_env import SO101ResidualEnv
from lerobot.policies.groot_base_policy import GR00TBasePolicy


def evaluate_groot_base(n_episodes=100, seed=42):
    """Evaluate GR00T base policy alone."""

    # Create environment with image observations
    env = SO101ResidualEnv(
        base_policy=None,  # No base policy (we'll use GR00T directly)
        use_image_obs=True,
        image_size=(224, 224),
        randomize=True,
        seed=seed,
    )

    # Load GR00T policy
    groot_policy = GR00TBasePolicy(
        model_path="phospho-app/gr00t-paper_return-7w9itxzsox",
        device="cuda",
        expected_action_dim=6,
    )

    # Evaluation loop
    successes = []
    distances = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0

        for step in range(400):
            # Get action from GR00T
            action = groot_policy.predict(obs)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        # Record results
        success = info.get("is_success", False)
        dist = info.get("dist_to_goal", 1.0)

        successes.append(success)
        distances.append(dist)

        print(f"Episode {episode+1}/{n_episodes}: "
              f"Success={success}, Distance={dist:.3f}m, Reward={episode_reward:.1f}")

    # Summary
    success_rate = np.mean(successes) * 100
    avg_distance = np.mean(distances)

    print("\n" + "="*80)
    print("GR00T Base Policy Evaluation Results")
    print("="*80)
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Avg Distance: {avg_distance:.3f}m")
    print()

    # Decision
    if success_rate < 5:
        print("❌ FAIL: Domain gap too large (<5% success)")
        print("   Recommendation: Use Jacobian IK instead")
    elif success_rate < 20:
        print("⚠️  CAUTION: Poor transfer (5-20% success)")
        print("   Recommendation: Consider Jacobian IK")
    else:
        print("✅ PASS: Good transfer (>20% success)")
        print("   Recommendation: Proceed with residual RL training")

    print("="*80)

    return success_rate, avg_distance


if __name__ == "__main__":
    evaluate_groot_base(n_episodes=100)
```

**Run Validation:**
```bash
python scripts/eval_groot_base.py

# Expected time: ~30-60 min (100 episodes with image rendering)

# Interpretation:
# - <5% success: DON'T use GR00T (domain gap too large)
# - 5-20% success: MAYBE use GR00T (marginal benefit)
# - >20% success: DO use GR00T (good transfer)
```

---

## Training Configuration

### Command Line

```bash
# After completing setup and validation:

source .venv/bin/activate

PYTHONPATH=src:$PYTHONPATH python src/lerobot/scripts/train_so101_residual.py \
    --base-policy il \
    --il-checkpoint phospho-app/gr00t-paper_return-7w9itxzsox \
    --alpha 0.5 \
    --total-timesteps 500000 \
    --n-envs 4 \
    --output-dir runs/groot_residual_improved_reward \
    --seed 42 \
    --learning-rate 3e-4 \
    --n-steps 256 \
    --batch-size 64
```

### Parameter Explanation

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--base-policy` | `il` | Use imitation learning (GR00T) as base |
| `--il-checkpoint` | `phospho-app/groot-paper_return-7w9itxzsox` | Fine-tuned GR00T model path |
| `--alpha` | `0.5` | 50% base + 50% residual (balanced) |
| `--total-timesteps` | `500000` | Same as Jacobian for fair comparison |
| `--n-envs` | `4` | 4 parallel environments (reduce from 8 due to image rendering) |
| `--learning-rate` | `3e-4` | Standard PPO learning rate |
| `--seed` | `42` | Reproducibility |

### Alpha Tuning

**Alpha (`α`) controls residual blending:**

```
total_action = base_action + α × residual_action
```

| Alpha | Base Weight | Residual Weight | Use Case |
|-------|-------------|-----------------|----------|
| 0.0 | 100% | 0% | Base policy only (evaluation) |
| 0.3 | 70% | 30% | Trust base policy more |
| 0.5 | 50% | 50% | **Balanced (recommended)** |
| 0.7 | 30% | 70% | More aggressive learning |
| 1.0 | 0% | 100% | Pure residual (like zero-policy) |

**Recommendation:** Start with `α=0.5`, tune based on base policy validation results:
- If base success >30%: Try `α=0.3` (trust base more)
- If base success 20-30%: Use `α=0.5` (balanced)
- If base success 10-20%: Try `α=0.7` (learn more corrections)

---

## Validation Protocol

### Pre-Training Validation

**Purpose:** Verify GR00T works in simulation before investing training time.

**Steps:**
1. Load GR00T model ✅
2. Test action extraction ✅
3. Run 100 episodes (α=0, base only) 🔍
4. Measure success rate 📊
5. Decide whether to proceed ✅/❌

**Script:** `scripts/eval_groot_base.py` (see Setup Instructions)

**Decision Criteria:**
```
if success_rate < 5%:
    print("❌ Don't use GR00T - domain gap too large")
    action = "Use Jacobian IK instead"
elif success_rate < 20%:
    print("⚠️  Marginal benefit - consider alternatives")
    action = "Evaluate cost/benefit of GR00T setup"
else:
    print("✅ Proceed with GR00T training")
    action = "Start residual RL training"
```

### Post-Training Validation

**Purpose:** Evaluate trained residual RL agent.

**Metrics:**
- Success rate (% of episodes where paper is in target)
- Average distance to goal (meters)
- Episode reward (cumulative)
- Residual magnitude (L2 norm of corrections)

**Comparison Baseline:**
| Metric | Jacobian Base | GR00T Base (Expected) |
|--------|---------------|----------------------|
| Base Success | 30-50% | 0-50% (validate first) |
| Final Success | 70-85% | 70-85% (if base >20%) |
| Training Time | 23 min | 2-3 hours |
| Domain Gap | None | Large |

---

## Expected Outcomes

### Best Case Scenario

**If GR00T base policy shows >30% success in validation:**

- Base policy provides task-specific knowledge
- Residual RL corrects errors and refines behavior
- **Final success rate:** 75-90%
- **Training time:** 2-3 hours
- **Benefit over Jacobian:** Slightly better final performance (5-10% improvement)

### Likely Scenario

**If GR00T base policy shows 10-30% success:**

- Partial transfer from real to sim
- Residual RL compensates for domain gap
- **Final success rate:** 65-80%
- **Training time:** 2-3 hours
- **Benefit over Jacobian:** Similar or slightly worse performance

### Worst Case Scenario

**If GR00T base policy shows <10% success:**

- Domain gap too large
- Random or poor actions from GR00T
- Residual RL must learn from scratch (like zero-policy)
- **Final success rate:** 40-60% (worse than Jacobian)
- **Training time:** 2-3 hours (wasted)
- **Recommendation:** Abort and use Jacobian instead

---

## Risk Assessment

### High Risk: Domain Gap (80% probability)

**Impact:** GR00T performs poorly in simulation

**Mitigation:**
- Run validation before training (100 episodes)
- Set success threshold (>20% to proceed)
- Have Jacobian baseline as fallback

**Contingency:**
- If validation fails: Use Jacobian IK
- If training fails: Fall back to completed Jacobian results

### Medium Risk: Integration Bugs (40% probability)

**Impact:** Image observation, model loading, or wrapper issues

**Mitigation:**
- Test each component individually
- Use existing test scripts
- Debug incrementally

**Contingency:**
- Allocate 2-4 hours for debugging
- Have Jacobian baseline as fallback

### Low Risk: Training Instability (20% probability)

**Impact:** Poor convergence, NaN losses, or low final performance

**Mitigation:**
- Start with validated hyperparameters (from Jacobian)
- Monitor training closely
- Use lower learning rate if needed

**Contingency:**
- Tune hyperparameters (alpha, learning rate)
- Extend training time
- Fall back to Jacobian if unstable

### Overall Risk Level: **HIGH**

**Recommendation:** Only proceed if:
1. You have 6-11 hours available
2. Planning real robot deployment (GR00T works better on real robot)
3. Interested in vision-based policies for research
4. Jacobian baseline is insufficient (<70% success)

Otherwise: **Use Jacobian IK baseline** (currently training, expected 70-85% success).

---

## Appendix A: Troubleshooting

### Issue: Model Loading Fails

```
ValueError: Transformers does not recognize this architecture
```

**Solution:**
- Install Isaac-GR00T package (Option A)
- OR use Phosphobot server (Option B)
- OR wait for transformers support (Option C)

### Issue: Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce `--n-envs` from 4 to 2 or 1
- Reduce `image_size` from (224, 224) to (128, 128)
- Use smaller batch size: `--batch-size 32`
- Clear CUDA cache: `torch.cuda.empty_cache()`

### Issue: Slow Training (<50 FPS)

**Causes:**
- Image rendering overhead
- Large batch size
- Too many parallel environments

**Solutions:**
- Accept slower training (expected with images)
- Reduce `--n-envs` to 2
- Reduce `image_size` to (128, 128)
- Use GPU for rendering if available

### Issue: Poor Performance (<60% success)

**Possible Causes:**
1. Domain gap too large (GR00T not transferring)
2. Wrong action convention (absolute vs relative)
3. Alpha too high/low
4. Insufficient training steps

**Solutions:**
1. Check validation results (should be >20%)
2. Verify action convention in `groot_base_policy.py`
3. Tune alpha (try 0.3, 0.5, 0.7)
4. Extend training to 1M steps

---

## Appendix B: File Manifest

**Existing Files:**
- `src/lerobot/policies/groot_base_policy.py` - GR00T wrapper ✅
- `scripts/test_groot_inference.py` - Model loading test ✅
- `docs/GROOT_INTEGRATION.md` - Integration docs ✅

**New Files (Need to Create):**
- `src/lerobot/envs/so101_groot_wrapper.py` - Environment wrapper
- `scripts/eval_groot_base.py` - Validation script
- `docs/GROOT_TRAINING_PLAN.md` - This document ✅

**Modified Files:**
- `src/lerobot/envs/so101_residual_env.py` - Add image observation support
- `src/lerobot/scripts/train_so101_residual.py` - Already supports IL base ✅

---

## Appendix C: References

**Model:**
- HuggingFace: https://huggingface.co/phospho-app/gr00t-paper_return-7w9itxzsox
- Dataset: https://huggingface.co/datasets/Hafnium49/paper_return

**Repositories:**
- Isaac-GR00T: https://github.com/NVIDIA/Isaac-GR00T
- Phosphobot: https://github.com/phospho-app/phosphobot

**Documentation:**
- GR00T Blog: https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning
- Seeed Wiki: https://wiki.seeedstudio.com/control_robotic_arm_via_gr00t/
- LeRobot Docs: https://huggingface.co/docs/lerobot

**Related Documents:**
- [GROOT_INTEGRATION.md](GROOT_INTEGRATION.md) - Initial research
- [TRAINING_GUIDE.md](../TRAINING_GUIDE.md) - General training guide
- [IMPLEMENTATION_PROGRESS.md](../IMPLEMENTATION_PROGRESS.md) - Project status

---

**Document Version:** 1.0
**Last Updated:** 2025-10-22
**Status:** BLOCKED - Awaiting Infrastructure Setup
**Next Review:** After Jacobian training completes (in ~10 minutes)
**Author:** Claude Code
**Maintainer:** Hafnium49 / lohpaul9
