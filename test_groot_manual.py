#!/usr/bin/env python3
"""
Manual test script for GR00T model loading after Phase 1 installation.

This script tests:
1. Isaac GR00T package import
2. Model loading from HuggingFace
3. Processor loading
4. Dummy inference
5. Action extraction

Run this after completing Phase 1 to verify installation.
"""

import sys
from pathlib import Path

import numpy as np
import torch

print("=" * 80)
print("GR00T Model Loading Test (Phase 1 Verification)")
print("=" * 80)
print()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

################################################################################
# Test 1: Import Isaac GR00T
################################################################################

print("Test 1: Importing Isaac GR00T package")
print("-" * 80)

try:
    import isaac_groot
    print("✅ isaac_groot imported successfully")
    print(f"   Version: {isaac_groot.__version__ if hasattr(isaac_groot, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"❌ Failed to import isaac_groot: {e}")
    print("\nTroubleshooting:")
    print("  1. Make sure you ran: pip install -e .[base] in Isaac-GR00T directory")
    print("  2. Check venv is activated: source .venv/bin/activate")
    print("  3. Verify installation: pip list | grep isaac")
    sys.exit(1)

print()

################################################################################
# Test 2: Check CUDA
################################################################################

print("Test 2: Checking CUDA availability")
print("-" * 80)

cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    device = "cuda"
else:
    print("⚠️  CUDA not available, using CPU (will be slow)")
    device = "cpu"

print()

################################################################################
# Test 3: Load GR00T Model
################################################################################

print("Test 3: Loading GR00T model from HuggingFace")
print("-" * 80)

model_path = "phospho-app/gr00t-paper_return-7w9itxzsox"
print(f"Model: {model_path}")
print("This may take a few minutes on first run (downloading ~2-4 GB)...")
print()

try:
    from transformers import AutoModel, AutoProcessor

    # Load model
    print("Loading model...")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,  # Required for custom architectures
    )
    model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully")
    print(f"   Device: {device}")
    print(f"   Model type: {type(model).__name__}")

    # Load processor
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    print(f"✅ Processor loaded successfully")

except Exception as e:
    print(f"❌ Failed to load model: {e}")
    print("\nTroubleshooting:")
    print("  1. Check internet connection (model downloads from HuggingFace)")
    print("  2. Verify HuggingFace token if model is private: huggingface-cli login")
    print("  3. Check GPU memory: nvidia-smi")
    print("  4. Try CPU mode if GPU OOM: device='cpu'")
    sys.exit(1)

print()

################################################################################
# Test 4: Dummy Inference
################################################################################

print("Test 4: Running dummy inference")
print("-" * 80)

# Create dummy RGB image (top-view camera)
print("Creating dummy 224x224 RGB image...")
dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

try:
    # Preprocess image
    print("Preprocessing image...")
    inputs = processor(images=dummy_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    print("Running model inference...")
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"✅ Inference successful")
    print(f"   Output keys: {list(outputs.keys())}")

except Exception as e:
    print(f"❌ Inference failed: {e}")
    print("\nTroubleshooting:")
    print("  1. Check GPU memory: nvidia-smi")
    print("  2. Reduce batch size or image resolution")
    print("  3. Try CPU mode if GPU OOM")
    sys.exit(1)

print()

################################################################################
# Test 5: Action Extraction
################################################################################

print("Test 5: Extracting SO-101 actions")
print("-" * 80)

try:
    # Extract modality-based actions
    arm_action = outputs["action.single_arm"]  # (batch, horizon, 6)
    gripper_action = outputs["action.gripper"]  # (batch, horizon, 1)

    print(f"Arm action shape: {arm_action.shape}")
    print(f"Gripper action shape: {gripper_action.shape}")

    # Take first timestep from action horizon
    arm_action_t0 = arm_action[0, 0].cpu().numpy()  # (6,)
    gripper_action_t0 = gripper_action[0, 0].cpu().numpy()  # (1,)

    # Concatenate for SO-101 (6 DOF: 5 arm + 1 gripper)
    so101_action = np.concatenate([arm_action_t0[:5], gripper_action_t0])

    print(f"\n✅ Action extraction successful")
    print(f"   SO-101 action shape: {so101_action.shape}")
    print(f"   SO-101 action values: {so101_action}")
    print(f"   Action range: [{so101_action.min():.3f}, {so101_action.max():.3f}]")

    # Verify dimensions
    assert so101_action.shape == (6,), f"Expected (6,) action, got {so101_action.shape}"

except Exception as e:
    print(f"❌ Action extraction failed: {e}")
    print("\nTroubleshooting:")
    print("  1. Check model output format matches expected modalities")
    print("  2. Verify action.single_arm and action.gripper keys exist")
    print("  3. Check action horizon dimension (should be 16)")
    sys.exit(1)

print()

################################################################################
# Test 6: Memory Usage
################################################################################

print("Test 6: Checking memory usage")
print("-" * 80)

if cuda_available:
    memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
    memory_reserved = torch.cuda.memory_reserved(0) / 1024**3

    print(f"GPU memory allocated: {memory_allocated:.2f} GB")
    print(f"GPU memory reserved: {memory_reserved:.2f} GB")

    if memory_allocated > 10:
        print("⚠️  High memory usage - may need to reduce batch size or image resolution")
    else:
        print("✅ Memory usage OK")
else:
    print("Skipping GPU memory check (CPU mode)")

print()

################################################################################
# Summary
################################################################################

print("=" * 80)
print("Phase 1 Verification Complete ✅")
print("=" * 80)
print()
print("All tests passed! Isaac GR00T installation is working correctly.")
print()
print("Verified components:")
print("  ✅ Isaac GR00T package import")
print("  ✅ Model loading from HuggingFace")
print("  ✅ Processor loading")
print("  ✅ Dummy inference")
print("  ✅ Action extraction (6D for SO-101)")
print(f"  ✅ Memory usage ({memory_allocated:.2f} GB)")
print()
print("Ready for Phase 2: Environment modification for image observations")
print()
print("Next steps:")
print("  1. Modify SO101ResidualEnv to support image observations")
print("  2. Create GR00TResidualWrapper")
print("  3. Run 100-episode validation (Phase 3)")
print()
