#!/usr/bin/env python
"""
Test GR00T model loading and inspect output structure.

This script:
1. Loads the MuJoCo SO-101 world
2. Renders an image from the top-view camera
3. Loads the fine-tuned GR00T model
4. Runs inference and inspects output structure
5. Validates action dimensions and format
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import mujoco as mj

print("=" * 80)
print("GR00T Model Inference Test")
print("=" * 80)
print()

# Step 1: Load MuJoCo world
print("Step 1: Loading MuJoCo world...")
xml_path = Path(__file__).parent.parent / "src/lerobot/envs/so101_assets/paper_square_realistic.xml"

if not xml_path.exists():
    print(f"❌ World file not found: {xml_path}")
    sys.exit(1)

try:
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    print(f"✅ MuJoCo world loaded: {xml_path.name}")
except Exception as e:
    print(f"❌ Failed to load world: {e}")
    sys.exit(1)

print()

# Step 2: Render top-view camera
print("Step 2: Rendering top-view camera...")
try:
    # GR00T expects 224x224 images
    renderer = mj.Renderer(model, height=224, width=224)
    cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "top_view")

    if cam_id < 0:
        print("❌ Camera 'top_view' not found in world")
        print("Available cameras:", [model.camera(i).name for i in range(model.ncam)])
        sys.exit(1)

    renderer.update_scene(data, camera=cam_id)
    image = renderer.render()  # (224, 224, 3) uint8

    print(f"✅ Image rendered: {image.shape}, dtype={image.dtype}, range=[{image.min()}, {image.max()}]")
except Exception as e:
    print(f"❌ Failed to render camera: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 3: Load GR00T model
print("Step 3: Loading GR00T base policy...")
print("   Model: phospho-app/gr00t-paper_return-7w9itxzsox")
print("   This may take a few minutes on first run (downloading model)...")
print()

try:
    from lerobot.policies.groot_base_policy import GR00TBasePolicy

    policy = GR00TBasePolicy(
        model_path="phospho-app/gr00t-paper_return-7w9itxzsox",
        device="cuda",
        expected_action_dim=6,
        use_first_timestep=True,
        action_convention="absolute",
        invert_gripper=True,
    )

    print("✅ GR00T policy loaded successfully")
    print(f"   Device: {policy.device}")
    print(f"   Expected action dim: {policy.expected_action_dim}")
    print(f"   Action convention: {policy.action_convention}")

except ImportError as e:
    print(f"❌ Failed to import GR00TBasePolicy: {e}")
    print("   Make sure transformers is installed: pip install transformers")
    sys.exit(1)
except Exception as e:
    print(f"❌ Failed to load GR00T model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 4: Run inference
print("Step 4: Running inference...")
try:
    action = policy.predict(image)

    print("✅ Inference successful")
    print(f"   Action shape: {action.shape}")
    print(f"   Action dtype: {action.dtype}")
    print(f"   Action values: {action}")
    print(f"   Action range: [{action.min():.4f}, {action.max():.4f}]")
    print(f"   Action mean: {action.mean():.4f}")
    print(f"   Action std: {action.std():.4f}")

    # Check for non-zero dimensions
    non_zero_dims = np.where(np.abs(action) > 0.01)[0]
    print(f"   Non-zero dims (>0.01): {non_zero_dims.tolist()}")

except Exception as e:
    print(f"❌ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 5: Validate action format
print("Step 5: Validating action format...")

validation_passed = True

# Check shape
if action.shape != (6,):
    print(f"⚠️  Expected action shape (6,), got {action.shape}")
    validation_passed = False
else:
    print("✅ Action shape correct: (6,)")

# Check dtype
if action.dtype != np.float32:
    print(f"⚠️  Expected dtype float32, got {action.dtype}")
    validation_passed = False
else:
    print("✅ Action dtype correct: float32")

# Check for NaN/Inf
if np.any(np.isnan(action)) or np.any(np.isinf(action)):
    print("❌ Action contains NaN or Inf values")
    validation_passed = False
else:
    print("✅ No NaN/Inf values")

# Check reasonable range (joint actions typically in [-3, 3] radians)
if np.any(np.abs(action) > 10.0):
    print(f"⚠️  Action values unusually large: max={np.abs(action).max():.2f}")
    print("   (May need normalization or scaling)")
else:
    print("✅ Action values in reasonable range")

print()

# Step 6: Test multiple inferences
print("Step 6: Testing multiple inferences for consistency...")
try:
    actions = []
    for i in range(5):
        a = policy.predict(image)
        actions.append(a)

    actions = np.array(actions)  # (5, 6)

    # Check consistency (should be identical for same image)
    consistency = np.std(actions, axis=0)
    max_std = consistency.max()

    print(f"   Consistency (std across 5 runs): max={max_std:.6f}")

    if max_std < 1e-5:
        print("✅ Deterministic inference (good)")
    elif max_std < 0.01:
        print("⚠️  Slight variation in outputs (may be due to sampling)")
    else:
        print("⚠️  High variation in outputs (unexpected)")

except Exception as e:
    print(f"❌ Multiple inference test failed: {e}")
    validation_passed = False

print()

# Summary
print("=" * 80)
if validation_passed:
    print("✅ ALL TESTS PASSED")
    print()
    print("GR00T base policy is ready to integrate with SO101ResidualEnv!")
    print()
    print("Next steps:")
    print("  1. Integrate with environment: modify so101_residual_env.py")
    print("  2. Test in environment: run validation script")
    print("  3. Train with residual RL: python scripts/train_ppo_residual.py --use-groot")
else:
    print("⚠️  SOME TESTS FAILED")
    print()
    print("Review the warnings above before proceeding.")
    print("The policy may still work, but double-check action ranges and conventions.")

print("=" * 80)
