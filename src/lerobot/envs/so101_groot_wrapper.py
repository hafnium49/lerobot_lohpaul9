#!/usr/bin/env python
"""
GR00T Residual RL Environment Wrapper.

This wrapper enables using a fine-tuned GR00T N1.5 model as the base policy
for residual reinforcement learning. It handles:
- Providing RGB images to GR00T base policy
- Providing state observations to residual RL policy
- Action blending (base + residual)
"""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class GR00TResidualWrapper(gym.Wrapper):
    """
    Wrapper for using GR00T as base policy with residual RL.

    The environment must support image observations (use_image_obs=True).
    This wrapper:
    1. Gets RGB image from environment
    2. Passes image to GR00T to get base action
    3. Adds residual action from RL policy
    4. Steps environment with combined action
    5. Returns state observation (not image) to RL policy

    Args:
        env: SO101ResidualEnv with use_image_obs=True
        groot_model_path: Path to fine-tuned GR00T model
        alpha: Residual blending factor (0=base only, 1=full residual)
        device: Device for GR00T inference ('cuda' or 'cpu')
    """

    def __init__(
        self,
        env: gym.Env,
        groot_model_path: str = "phospho-app/gr00t-paper_return-7w9itxzsox",
        alpha: float = 0.5,
        device: str = "auto",
    ):
        """Initialize GR00T residual wrapper."""
        super().__init__(env)

        # Check that environment supports image observations
        if not hasattr(env, 'use_image_obs') or not env.use_image_obs:
            raise ValueError(
                "Environment must have use_image_obs=True for GR00T wrapper. "
                "Create environment with: SO101ResidualEnv(use_image_obs=True)"
            )

        # Load GR00T base policy
        logger.info(f"Loading GR00T base policy from {groot_model_path}")
        from lerobot.policies.groot_base_policy import GR00TBasePolicy

        self.groot_policy = GR00TBasePolicy(
            model_path=groot_model_path,
            device=device,
            expected_action_dim=6,  # SO-101: 5 arm + 1 gripper
        )

        self.alpha = alpha

        # Override observation space to state-only for RL policy
        # (GR00T uses images internally, but RL policy sees state)
        state_dim = env.observation_space["state"].shape[0]
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

        logger.info(f"✅ GR00T wrapper initialized (alpha={alpha})")

    def step(self, residual_action: np.ndarray):
        """
        Step environment with GR00T base + residual action.

        Args:
            residual_action: Residual action from RL policy (6D)

        Returns:
            state_obs: State observation for RL policy (not image)
            reward: Reward signal
            terminated: Episode terminated flag
            truncated: Episode truncated flag
            info: Info dict with base and total actions
        """
        # Get current observation (includes image)
        # Note: We need to manually get the observation since we haven't stepped yet
        # Use the environment's current state
        current_obs = self.env._get_obs()

        # Extract image for GR00T
        image = current_obs["image"]

        # Get base action from GR00T
        base_action = self.groot_policy.predict(image)

        # Blend actions: total = base + alpha * residual
        total_action = base_action + self.alpha * residual_action

        # Step environment with blended action
        obs, reward, terminated, truncated, info = self.env.step(total_action)

        # Extract state observation for RL policy
        state_obs = obs["state"]

        # Add action info to info dict
        info["base_action"] = base_action
        info["residual_action"] = residual_action
        info["total_action"] = total_action

        return state_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment and GR00T policy state."""
        obs, info = self.env.reset(**kwargs)

        # Reset GR00T policy state
        self.groot_policy.reset()

        # Extract state observation for RL policy
        state_obs = obs["state"]

        return state_obs, info

    def render(self):
        """Render environment."""
        return self.env.render()


# Test script
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from lerobot.envs.so101_residual_env import SO101ResidualEnv

    print("=" * 80)
    print("GR00T Residual Wrapper - Standalone Test")
    print("=" * 80)
    print()

    try:
        # Create environment with image observations
        print("Creating SO101ResidualEnv with image observations...")
        env = SO101ResidualEnv(
            use_image_obs=True,
            image_size=(224, 224),
            camera_name_for_obs="top_view",
        )
        print("✅ Environment created")
        print(f"   Observation space: {env.observation_space}")
        print()

        # Wrap with GR00T wrapper
        print("Wrapping with GR00T residual wrapper...")
        wrapped_env = GR00TResidualWrapper(
            env,
            groot_model_path="phospho-app/gr00t-paper_return-7w9itxzsox",
            alpha=0.5,
        )
        print("✅ Wrapper created")
        print(f"   Observation space (RL policy): {wrapped_env.observation_space}")
        print()

        # Test reset
        print("Testing reset...")
        obs, info = wrapped_env.reset()
        print(f"✅ Reset successful")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Observation type: {obs.dtype}")
        print()

        # Test step
        print("Testing step with random residual action...")
        residual_action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(residual_action)

        print(f"✅ Step successful")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Reward: {reward:.3f}")
        print(f"   Base action: {info['base_action']}")
        print(f"   Residual action: {info['residual_action']}")
        print(f"   Total action: {info['total_action']}")
        print()

        print("=" * 80)
        print("✅ Standalone test PASSED")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
