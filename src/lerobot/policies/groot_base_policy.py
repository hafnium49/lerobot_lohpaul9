#!/usr/bin/env python
"""
GR00T N1.5 Base Policy Wrapper for Residual RL.

This module provides a wrapper for fine-tuned GR00T N1.5 models to use as
base policies in residual reinforcement learning. The wrapper handles:
- Image preprocessing for GR00T's vision backbone
- Modality-based action extraction (single_arm + gripper)
- Action convention conversion (absolute positions → deltas if needed)
- Action horizon handling (16-step sequences → single action)

The GR00T model outputs actions as a modality dictionary:
{
    "action.single_arm": (16, 6),  # 16 timesteps, 6 arm joints
    "action.gripper": (16, 1),     # 16 timesteps, 1 gripper DOF
}

For SO-101 robot with 6 total DOF (5 arm + 1 gripper), we extract the
relevant dimensions and optionally convert to delta actions.
"""

import numpy as np
import torch
from typing import Union, Optional, Dict, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)


class GR00TBasePolicy:
    """
    Wrapper for fine-tuned GR00T N1.5 model as base policy.

    This class loads a fine-tuned GR00T model from HuggingFace and provides
    a simple interface for getting base actions from RGB images.

    Args:
        model_path: HuggingFace model path (default: phospho-app/gr00t-paper_return-7w9itxzsox)
        device: Device to run inference on ('cuda' or 'cpu')
        use_first_timestep: Whether to use first timestep of 16-step horizon
        action_convention: 'absolute' (joint positions) or 'relative' (joint deltas)
        expected_action_dim: Expected output dimension (6 or 7 for SO-101)
        invert_gripper: Whether to invert gripper polarity (as per Seeed Wiki)
    """

    def __init__(
        self,
        model_path: str = "phospho-app/gr00t-paper_return-7w9itxzsox",
        device: Optional[str] = None,
        use_first_timestep: bool = True,
        action_convention: str = "absolute",
        expected_action_dim: int = 6,
        invert_gripper: bool = True,
    ):
        """Initialize GR00T base policy."""

        # Device selection
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Configuration
        self.model_path = model_path
        self.use_first_timestep = use_first_timestep
        self.action_convention = action_convention
        self.expected_action_dim = expected_action_dim
        self.invert_gripper = invert_gripper

        # State tracking for delta conversion
        self.prev_action = None

        # Load model and processor
        logger.info(f"Loading GR00T model from {model_path}")
        logger.info(f"Device: {device}")

        try:
            from gr00t.model.policy import Gr00tPolicy
            from gr00t.experiment.data_config import BaseDataConfig, DATA_CONFIG_MAP
            from gr00t.data.dataset import ModalityConfig
            from gr00t.data.transform.base import ComposedModalityTransform
            from gr00t.data.transform.concat import ConcatTransform
            from gr00t.data.transform.state_action import (
                StateActionToTensor,
                StateActionTransform,
            )
            from gr00t.data.transform.video import (
                VideoColorJitter,
                VideoCrop,
                VideoResize,
                VideoToNumpy,
                VideoToTensor,
            )
            from gr00t.model.transforms import GR00TTransform

            # Create custom data config for fine-tuned model
            # Model was trained with 'video.image_cam_0' and 'video.image_cam_1' keys
            class FineTunedSO101DataConfig(BaseDataConfig):
                video_keys = ["video.image_cam_0", "video.image_cam_1"]
                state_keys = ["state.arm_0"]
                action_keys = ["action.arm_0"]
                language_keys = []
                observation_indices = [0]
                action_indices = list(range(16))

                def transform(self):
                    # Eval-only transforms (no augmentation for inference)
                    transforms = [
                        # video transforms
                        VideoToTensor(apply_to=self.video_keys),
                        VideoCrop(apply_to=self.video_keys, scale=0.95),
                        VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
                        # Skip VideoColorJitter for eval (augmentation only for training)
                        VideoToNumpy(apply_to=self.video_keys),
                        # state transforms (even though we won't use them for inference)
                        StateActionToTensor(apply_to=self.state_keys),
                        StateActionTransform(
                            apply_to=self.state_keys,
                            normalization_modes={key: "min_max" for key in self.state_keys},
                        ),
                        # action transforms (for output processing)
                        StateActionToTensor(apply_to=self.action_keys),
                        StateActionTransform(
                            apply_to=self.action_keys,
                            normalization_modes={key: "min_max" for key in self.action_keys},
                        ),
                        # concat transforms
                        ConcatTransform(
                            video_concat_order=self.video_keys,
                            state_concat_order=self.state_keys,
                            action_concat_order=self.action_keys,
                        ),
                        # model-specific transform (adds image_sizes, etc.)
                        GR00TTransform(
                            state_horizon=len(self.observation_indices),
                            action_horizon=len(self.action_indices),
                            max_state_dim=64,
                            max_action_dim=32,
                        ),
                    ]
                    return ComposedModalityTransform(transforms=transforms)

            # Use custom config matching fine-tuned model's training data
            data_config = FineTunedSO101DataConfig()
            modality_config = data_config.modality_config()
            modality_transform = data_config.transform()

            # Load GR00T policy
            # Use 'new_embodiment' tag as SO-101 is a custom fine-tuned model
            self.policy = Gr00tPolicy(
                model_path=model_path,
                embodiment_tag="new_embodiment",  # Custom embodiment tag
                modality_config=modality_config,
                modality_transform=modality_transform,
                device=device,
            )

            logger.info("✅ GR00T policy loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load GR00T policy: {e}")
            logger.error("Note: This requires Isaac GR00T package to be installed")
            logger.error("Install with: cd ~/Isaac-GR00T && pip install -e .[base]")
            raise

    def predict(self, image: np.ndarray, current_qpos: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get base policy action from RGB image.

        Args:
            image: RGB image (H, W, 3), numpy array, values in [0, 255]
            current_qpos: Current joint positions (optional, needed for absolute→relative conversion)

        Returns:
            action: (6,) or (7,) numpy array - joint deltas or positions

        Raises:
            ValueError: If action dimensions don't match expected format
        """

        # Prepare observation dict for GR00T policy
        # Fine-tuned model expects specific camera keys from training dataset
        try:
            # Create observation dict matching model's expected format
            # Model was trained with 'video.image_cam_0' and 'video.image_cam_1' keys
            # We provide the same image for both cameras (single-camera setup)
            # Also need dummy state and action for transforms
            obs_dict = {
                "video.image_cam_0": image[np.newaxis, ...],  # Add batch dim: (1, H, W, 3)
                "video.image_cam_1": image[np.newaxis, ...],  # Duplicate for second camera
                "state.arm_0": np.zeros((1, 6), dtype=np.float32),  # Dummy state
                "action.arm_0": np.zeros((1, 6), dtype=np.float32),  # Dummy action
            }

            # Get action from policy
            outputs = self.policy.get_action(obs_dict)

        except Exception as e:
            logger.error(f"GR00T inference failed: {e}")
            raise

        # Extract actions from modality-based output
        action = self._extract_action_from_outputs(outputs)

        # Convert absolute → relative if needed
        if self.action_convention == "absolute" and current_qpos is not None:
            action = action - current_qpos

        # Track for future delta conversion
        if self.action_convention == "absolute":
            self.prev_action = action.copy()

        return action.astype(np.float32)

    def _extract_action_from_outputs(
        self,
        outputs: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> np.ndarray:
        """
        Extract action array from GR00T model outputs.

        GR00T returns actions as modality dictionary:
        {
            "action.single_arm": (batch, 16, 6),
            "action.gripper": (batch, 16, 1),
        }

        We extract the first timestep and concatenate modalities.

        Args:
            outputs: Model outputs (dict or tensor)

        Returns:
            action: (6,) or (7,) numpy array
        """

        # Case 1: Modality dictionary (expected format)
        if isinstance(outputs, dict):
            try:
                # Check if it's the fine-tuned model format (action.arm_0)
                if "action.arm_0" in outputs:
                    # Fine-tuned model format: action.arm_0 contains full action
                    full_action = outputs["action.arm_0"]

                    # Convert to numpy
                    if isinstance(full_action, torch.Tensor):
                        full_action = full_action.cpu().numpy()

                    # Handle batch and timestep dimensions
                    # Expected shape: (batch, 16, 6)
                    if full_action.ndim == 3:
                        full_action = full_action[0]  # Remove batch dim → (16, 6)

                    # Use first timestep if action horizon
                    if self.use_first_timestep and full_action.ndim == 2:
                        full_action = full_action[0]  # (16, 6) → (6,)

                    logger.debug(f"Full action shape: {full_action.shape}")
                    logger.debug(f"Full action values: {full_action}")

                    # Return the action as-is (already 6D for SO-101)
                    action = full_action[:self.expected_action_dim]

                    # Invert gripper polarity if needed
                    if self.invert_gripper:
                        action[-1] = action[-1] * -1

                    return action

                # Standard GR00T format: separate arm and gripper
                arm_action = outputs.get("action.single_arm")
                gripper_action = outputs.get("action.gripper")

                if arm_action is None or gripper_action is None:
                    raise KeyError("Expected 'action.single_arm' and 'action.gripper' in outputs")

                # Convert to numpy
                if isinstance(arm_action, torch.Tensor):
                    arm_action = arm_action.cpu().numpy()
                if isinstance(gripper_action, torch.Tensor):
                    gripper_action = gripper_action.cpu().numpy()

                # Handle batch and timestep dimensions
                # Expected shapes: (batch, 16, 6) and (batch, 16, 1)
                if arm_action.ndim == 3:
                    arm_action = arm_action[0]  # Remove batch dim → (16, 6)
                if gripper_action.ndim == 3:
                    gripper_action = gripper_action[0]  # Remove batch dim → (16, 1)

                # Use first timestep if action horizon
                if self.use_first_timestep and arm_action.ndim == 2:
                    arm_action = arm_action[0]  # (16, 6) → (6,)
                    gripper_action = gripper_action[0]  # (16, 1) → (1,)

                # Flatten gripper if needed
                if gripper_action.ndim > 1:
                    gripper_action = gripper_action.flatten()

                logger.debug(f"Arm action shape: {arm_action.shape}")
                logger.debug(f"Gripper action shape: {gripper_action.shape}")

                # Handle dimension mapping
                if self.expected_action_dim == 6:
                    # SO-101 with 6 total DOF: 5 arm + 1 gripper
                    # GR00T returns 6 arm dims, we take first 5
                    if arm_action.shape[0] >= 5:
                        action = np.concatenate([arm_action[:5], gripper_action[:1]])
                    else:
                        raise ValueError(f"Expected at least 5 arm dims, got {arm_action.shape[0]}")

                elif self.expected_action_dim == 7:
                    # SO-101 with 7 total DOF: 6 arm + 1 gripper separate
                    action = np.concatenate([arm_action[:6], gripper_action[:1]])

                else:
                    raise ValueError(f"Unsupported action dim: {self.expected_action_dim}")

                # Invert gripper polarity if needed (as per Seeed Wiki)
                if self.invert_gripper:
                    action[-1] = action[-1] * -1

                logger.debug(f"Final action: {action}")
                return action

            except Exception as e:
                logger.error(f"Failed to extract actions from modality dict: {e}")
                logger.error(f"Output keys: {outputs.keys() if isinstance(outputs, dict) else 'N/A'}")
                raise

        # Case 2: Raw tensor output (fallback)
        elif isinstance(outputs, torch.Tensor):
            logger.warning("Received raw tensor output instead of modality dict")

            outputs_np = outputs.cpu().numpy()

            # Handle batch and timestep dimensions
            if outputs_np.ndim == 3:  # (batch, timesteps, dims)
                outputs_np = outputs_np[0, 0, :]  # Take first batch, first timestep
            elif outputs_np.ndim == 2:  # (timesteps, dims)
                outputs_np = outputs_np[0, :]  # Take first timestep

            # Extract expected dimensions
            if outputs_np.shape[0] >= self.expected_action_dim:
                action = outputs_np[:self.expected_action_dim]
            else:
                raise ValueError(f"Output has {outputs_np.shape[0]} dims, expected {self.expected_action_dim}")

            # Invert gripper
            if self.invert_gripper:
                action[-1] = action[-1] * -1

            return action

        else:
            raise TypeError(f"Unexpected output type: {type(outputs)}")

    def reset(self):
        """Reset internal state (for delta conversion)."""
        self.prev_action = None

    def __call__(self, image: np.ndarray, current_qpos: Optional[np.ndarray] = None) -> np.ndarray:
        """Callable interface for compatibility."""
        return self.predict(image, current_qpos)


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path

    print("=" * 80)
    print("GR00T Base Policy - Standalone Test")
    print("=" * 80)
    print()

    # Create dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    try:
        # Load policy
        print("Loading GR00T base policy...")
        policy = GR00TBasePolicy(
            model_path="phospho-app/gr00t-paper_return-7w9itxzsox",
            device="cuda" if torch.cuda.is_available() else "cpu",
            expected_action_dim=6,
        )
        print("✅ Policy loaded")
        print()

        # Test inference
        print("Testing inference on dummy image...")
        action = policy.predict(dummy_image)

        print(f"✅ Inference successful")
        print(f"   Action shape: {action.shape}")
        print(f"   Action values: {action}")
        print(f"   Action range: [{action.min():.3f}, {action.max():.3f}]")
        print()
        print("=" * 80)
        print("✅ Standalone test PASSED")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
