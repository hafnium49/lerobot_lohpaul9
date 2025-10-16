# GR00T N1.5 Integration Documentation

**Date:** 2025-10-16
**Status:** Implementation Ready - Awaiting Isaac-GR00T Package Setup

---

## Summary

We've successfully researched and planned the integration of the fine-tuned GR00T N1.5 model (phospho-app/gr00t-paper_return-7w9itxzsox) as a base policy for residual RL. The model is trained on the exact task dataset (Hafnium49/paper_return) and uses a modality-based action output format.

---

## Key Findings

### 1. Action Dimension Mapping (RESOLVED)

**GR00T Output Format:**
- **NOT raw 32D arrays** as initially thought
- Returns **modality dictionary**:
  ```python
  {
      "action.single_arm": (16, 6),  # 16 timesteps, 6 arm joints
      "action.gripper": (16, 1),     # 16 timesteps, 1 gripper DOF
  }
  ```

**SO-101 Action Mapping:**
- **6 DOF total:** 5 arm joints + 1 gripper
- **Extraction:**
  ```python
  arm_action = outputs["action.single_arm"][0, 0]  # First timestep → (6,)
  gripper_action = outputs["action.gripper"][0, 0]  # First timestep → (1,)

  # For our 6 DOF env:
  action = np.concatenate([arm_action[:5], gripper_action])  # (6,)
  ```

**The 32D `max_action_dim`:**
- Internal padding parameter in GR00T's diffusion architecture
- Supports multi-embodiment learning
- **We don't interact with it directly during inference**

### 2. Model Loading Requirements

**Challenge:**
- GR00T N1.5 is a **custom architecture** (`model_type: "gr00t_n1_5"`)
- **NOT available** in standard `transformers` library
- Cannot use `AutoModel.from_pretrained()` directly

**Required Setup:**
- Install NVIDIA Isaac-GR00T package
- OR use Phosphobot inference server (client-server architecture)
- OR implement custom model loading code

**Error Encountered:**
```
ValueError: The checkpoint you are trying to load has model type `gr00t_n1_5`
but Transformers does not recognize this architecture.
```

### 3. Action Convention

**From Seeed Wiki:**
- GR00T likely outputs **absolute joint positions**
- May need conversion to **relative deltas** for residual RL
- Gripper polarity may need inversion: `action[-1] *= -1`

---

## Files Created

### 1. `src/lerobot/policies/groot_base_policy.py` ✅

**Purpose:** Wrapper class for GR00T N1.5 model integration

**Features:**
- Modality-based action extraction
- Action dimension handling (6D or 7D)
- Absolute → relative conversion support
- Gripper polarity inversion
- Action horizon handling (16 timesteps → single action)

**Status:** Code complete, awaiting proper model loading

**Key Methods:**
```python
class GR00TBasePolicy:
    def __init__(model_path, device, expected_action_dim=6):
        # Load GR00T model (requires Isaac-GR00T package)

    def predict(image, current_qpos=None):
        # Get 6D action from RGB image
        # Returns: (6,) numpy array

    def _extract_action_from_outputs(outputs):
        # Parse modality dict → concatenate arm + gripper
```

### 2. `scripts/test_groot_inference.py` ✅

**Purpose:** Test script for model loading and action inspection

**Features:**
- Loads MuJoCo world
- Renders top-view camera
- Tests GR00T model loading
- Validates action format
- Checks determinism

**Status:** Ready to run once Isaac-GR00T is installed

---

## Integration Plan

### Current Status: Phase 3A (Partial)

**Completed:**
- ✅ Action mapping research and resolution
- ✅ GR00TBasePolicy wrapper implementation
- ✅ Test script creation
- ✅ Documentation

**Blocked:**
- ❌ Model loading (requires Isaac-GR00T package)
- ⏸️ Environment integration (depends on model loading)
- ⏸️ Training with GR00T base (depends on above)

### Next Steps

#### Option 1: Install Isaac-GR00T (Recommended for Full Integration)

**Steps:**
```bash
# Clone Isaac-GR00T repository
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T

# Install package
pip install -e .

# Test model loading
python scripts/test_groot_inference.py
```

**Time Estimate:** 2-3 hours (setup + testing)

#### Option 2: Use Phosphobot Inference Server

**Steps:**
```bash
# Install Phosphobot
pip install phosphobot

# Start inference server
python -m phosphobot.am.gr00t.serve --model-path phospho-app/gr00t-paper_return-7w9itxzsox

# Modify GR00TBasePolicy to use client API
```

**Time Estimate:** 3-4 hours (server setup + client integration)

#### Option 3: Proceed with Zero-Action Baseline First (Pragmatic)

**Rationale:**
- Establishes baseline performance immediately
- Validates residual RL pipeline works
- Provides comparison metrics for GR00T integration
- Gives time to properly set up infrastructure

**Steps:**
```bash
# Run full PPO training with zero-action base
python scripts/train_ppo_residual.py

# Expected: 20-40% success rate (no prior knowledge)
# Target: 85-90% with residual learning
```

**Time Estimate:** 2-4 hours (compute time)

**After baseline:** Return to GR00T integration with proper setup

---

## Technical Details

### Modality Configuration (from Research)

**SO-101 Dataset Format:**
```json
{
  "state": {
    "single_arm": {"start": 0, "end": 6},
    "gripper": {"start": 6, "end": 7}
  },
  "action": {
    "single_arm": {"start": 0, "end": 6},
    "gripper": {"start": 6, "end": 7}
  }
}
```

### GR00T Inference Flow

```python
# 1. Preprocess image
inputs = processor(images=image, return_tensors="pt")

# 2. Run model
outputs = model(**inputs)

# 3. Extract modalities
arm = outputs["action.single_arm"][0, 0]  # (6,)
gripper = outputs["action.gripper"][0, 0]  # (1,)

# 4. Concatenate for SO-101
action = np.concatenate([arm[:5], gripper])  # (6,)

# 5. Optional: invert gripper
action[-1] *= -1
```

### Expected Performance

**Base Policy Alone (α=0):**
- **Target:** 20-50% success rate
- **Indicates:** Task-specific knowledge from IL training

**Residual RL (α=0.7, 0.5-1M steps):**
- **Target:** 85-90% success rate
- **Improvement:** ~40-70% over base policy alone

### Sim-to-Real Considerations

**Potential Issue:**
- GR00T trained on **real SO-101 camera images**
- May not recognize **MuJoCo rendered images** (domain gap)

**Validation:**
- Test base policy alone for 100 episodes
- If success rate <5%: Domain gap too large
- If 5-20%: Marginal but may help residual
- If >20%: Good transfer, proceed

**Mitigation:**
- If poor transfer: Use zero-action baseline instead
- Document for future work (domain adaptation, Phase 8)

---

## Comparison: Zero-Action vs GR00T Base

| Aspect | Zero-Action | GR00T Base |
|--------|-------------|------------|
| **Setup Time** | 0 minutes | 2-4 hours |
| **Base Success** | 0% | 20-50% (expected) |
| **Sample Efficiency** | Lower | Higher |
| **Training Time** | Longer | Shorter |
| **Risk** | None | Sim-to-real gap |
| **Final Performance** | 85-90% (target) | 85-90% (target) |

**Key Insight:** Both should reach similar **final performance** with sufficient training. GR00T provides faster convergence but requires setup time.

---

## Recommendation

### Immediate Action: Run Zero-Action Baseline

**Rationale:**
1. **Validate pipeline:** Ensure residual RL training works end-to-end
2. **Establish metrics:** Baseline for comparing GR00T integration
3. **No blockers:** Can start immediately
4. **Lower risk:** No sim-to-real concerns

**Command:**
```bash
python scripts/train_ppo_residual.py
```

**Expected Results:**
- Training time: 2-4 hours (50k steps)
- Success rate: 40-70% (without strong prior)
- If <85%: Extend to 100k-500k steps

### Follow-Up: Integrate GR00T

**After baseline completes:**
1. Install Isaac-GR00T package
2. Test model loading with `test_groot_inference.py`
3. Validate base policy performance (100 episodes, α=0)
4. If >20% success: Run full residual RL training
5. Compare: GR00T convergence speed vs zero-action

---

## Resources

**Model:**
- HuggingFace: https://huggingface.co/phospho-app/gr00t-paper_return-7w9itxzsox
- Dataset: https://huggingface.co/datasets/Hafnium49/paper_return

**Repositories:**
- Isaac-GR00T: https://github.com/NVIDIA/Isaac-GR00T
- Phosphobot: https://github.com/phospho-app/phosphobot

**Documentation:**
- GR00T Blog: https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning
- Seeed Wiki: https://wiki.seeedstudio.com/control_robotic_arm_via_gr00t/

---

## Future Work (Phase 8)

**Vision-Based Residual RL:**
- Current: State-only observations (25D privileged info)
- Future: RGB observations for residual policy too
- Challenge: Sim-to-real for both base AND residual

**Domain Randomization:**
- Vary lighting, camera pose, table texture
- Improve sim-to-real transfer
- May allow GR00T to work better in sim

**Action Space Alignment:**
- Verify GR00T action convention (absolute vs relative)
- Add conversion layer if needed
- Test on real SO-101 hardware

---

**Last Updated:** 2025-10-16
**Status:** Ready for Zero-Action Baseline Training
**Next Milestone:** GR00T Integration (requires Isaac-GR00T setup)
