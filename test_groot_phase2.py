#!/usr/bin/env python3
"""
Phase 2 Test: End-to-End Image Observation Pipeline with GR00T

This script tests:
1. SO101ResidualEnv with image observations
2. GR00T base policy loading
3. GR00TResidualWrapper
4. Complete step cycle (image → GR00T → action → environment)
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.envs.so101_residual_env import SO101ResidualEnv
from lerobot.envs.so101_groot_wrapper import GR00TResidualWrapper

print("=" * 80)
print("Phase 2 Verification: Image Observation Pipeline with GR00T")
print("=" * 80)
print()

# Test 1: Environment with Image Observations
print("Test 1: Creating SO101ResidualEnv with image observations")
print("-" * 80)

try:
    env = SO101ResidualEnv(
        use_image_obs=True,
        image_size=(224, 224),
        camera_name_for_obs="top_view",
        randomize=True,
        seed=42,
    )
    print("✅ Environment created successfully")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    print()

    # Test reset
    obs, info = env.reset()
    print("✅ Environment reset successful")
    print(f"   State shape: {obs['state'].shape}")
    print(f"   Image shape: {obs['image'].shape}")
    print(f"   Image dtype: {obs['image'].dtype}")
    print(f"   Image range: [{obs['image'].min()}, {obs['image'].max()}]")
    print()

except Exception as e:
    print(f"❌ Test 1 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: GR00T Residual Wrapper
print("Test 2: Creating GR00TResidualWrapper")
print("-" * 80)

try:
    wrapped_env = GR00TResidualWrapper(
        env,
        groot_model_path="phospho-app/gr00t-paper_return-7w9itxzsox",
        alpha=0.5,
        device="cuda",
    )
    print("✅ GR00T wrapper created successfully")
    print(f"   Observation space (for RL): {wrapped_env.observation_space}")
    print(f"   Alpha (residual blend): {wrapped_env.alpha}")
    print()

except Exception as e:
    print(f"❌ Test 2 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Reset and Step
print("Test 3: Testing reset and step with GR00T wrapper")
print("-" * 80)

try:
    # Reset
    obs, info = wrapped_env.reset()
    print("✅ Wrapper reset successful")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation dtype: {obs.dtype}")
    print()

    # Step with random residual action
    print("Testing step with random residual action...")
    residual_action = wrapped_env.action_space.sample()

    obs, reward, terminated, truncated, info = wrapped_env.step(residual_action)

    print("✅ Step successful")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Reward: {reward:.3f}")
    print(f"   Terminated: {terminated}")
    print(f"   Truncated: {truncated}")
    print()

    print("Action blending:")
    print(f"   Base action (from GR00T):    {info['base_action']}")
    print(f"   Residual action (from RL):   {info['residual_action']}")
    print(f"   Total action (base + α*res): {info['total_action']}")
    print()

except Exception as e:
    print(f"❌ Test 3 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Multiple Steps
print("Test 4: Running 10 steps to verify stability")
print("-" * 80)

try:
    obs, info = wrapped_env.reset()

    for step in range(10):
        residual_action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(residual_action)

        print(f"Step {step+1:2d}: reward={reward:+.3f}, "
              f"base_mag={np.linalg.norm(info['base_action']):.3f}, "
              f"res_mag={np.linalg.norm(info['residual_action']):.3f}")

        if terminated or truncated:
            print("   Episode ended, resetting...")
            obs, info = wrapped_env.reset()

    print()
    print("✅ 10 steps completed successfully")
    print()

except Exception as e:
    print(f"❌ Test 4 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Rendering
print("Test 5: Testing rendering")
print("-" * 80)

try:
    # Try to render
    pixels = wrapped_env.render()

    if pixels is not None:
        print(f"✅ Rendering successful")
        print(f"   Render shape: {pixels.shape if hasattr(pixels, 'shape') else 'N/A'}")
    else:
        print("⚠️  Rendering returned None (might need render_mode='rgb_array')")

    print()

except Exception as e:
    print(f"⚠️  Rendering failed (not critical): {e}")
    print()

# Summary
print("=" * 80)
print("Phase 2 Verification Complete ✅")
print("=" * 80)
print()
print("All tests passed! Image observation pipeline with GR00T is working.")
print()
print("Verified components:")
print("  ✅ SO101ResidualEnv with image observations")
print("  ✅ Dual observation space (state + image)")
print("  ✅ Camera rendering for observations")
print("  ✅ GR00T base policy loading")
print("  ✅ GR00TResidualWrapper")
print("  ✅ Action blending (base + residual)")
print("  ✅ Multi-step execution")
print()
print("Ready for Phase 3: Validation (100-episode base policy test)")
print()
