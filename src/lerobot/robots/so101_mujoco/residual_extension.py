#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Residual action extension for SO101 MuJoCo robot.

This module extends the existing SO101MujocoRobot with residual action support,
enabling residual reinforcement learning on top of base policies.
"""

import logging
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ResidualActionMixin:
    """
    Mixin class that adds residual action support to SO101MujocoRobot.

    This can be mixed into the robot class to enable residual RL without
    modifying the core implementation.
    """

    def __init__(
        self,
        base_policy: Optional[Callable] = None,
        alpha: float = 0.5,
        residual_scale: float = 0.02,
        residual_mode: bool = False,
    ):
        """
        Initialize residual action support.

        Args:
            base_policy: Base policy that provides nominal actions
            alpha: Residual blending factor (0=base only, 1=full residual)
            residual_scale: Scaling factor for residual actions
            residual_mode: Whether to use residual mode
        """
        self.base_policy = base_policy
        self.alpha = alpha
        self.residual_scale = residual_scale
        self.residual_mode = residual_mode
        self._last_base_action = None
        self._last_residual_action = None

    def send_action_residual(self, residual_action: dict[str, Any]) -> dict[str, Any]:
        """
        Send residual action that gets added to base policy output.

        Args:
            residual_action: Residual corrections to add to base action
                            Should contain joint position deltas

        Returns:
            Dictionary with executed action and debug info
        """
        if not hasattr(self, 'is_connected') or not self.is_connected:
            raise RuntimeError("Robot must be connected")

        # Get current observation for base policy
        obs = self.get_observation()

        # Get base action from base policy
        base_action = {}
        if self.base_policy is not None:
            try:
                # Convert observation to format expected by base policy
                obs_array = self._obs_dict_to_array(obs)
                base_deltas = self.base_policy(obs_array)

                # Convert base deltas to action dict
                for i, joint_name in enumerate(self.JOINT_NAMES):
                    key = f"{joint_name}.pos"
                    # Base action is current position + base delta
                    current_pos = obs[f"{joint_name}.pos"]
                    base_action[key] = current_pos + base_deltas[i]

            except Exception as e:
                logger.warning(f"Base policy failed: {e}, using current positions")
                for joint_name in self.JOINT_NAMES:
                    key = f"{joint_name}.pos"
                    base_action[key] = obs[f"{joint_name}.pos"]
        else:
            # No base policy - use current positions
            for joint_name in self.JOINT_NAMES:
                key = f"{joint_name}.pos"
                base_action[key] = obs[f"{joint_name}.pos"]

        # Scale and apply residual
        total_action = {}
        for joint_name in self.JOINT_NAMES:
            key = f"{joint_name}.pos"

            # Get residual delta (scaled)
            residual_delta = residual_action.get(key, 0.0) * self.residual_scale

            # Combine base and residual
            total_action[key] = base_action[key] + self.alpha * residual_delta

            # Clip to joint limits
            dof_id = self.dof_ids[joint_name]
            if hasattr(self, 'j_lo') and hasattr(self, 'j_hi'):
                total_action[key] = np.clip(
                    total_action[key],
                    self.j_lo[dof_id],
                    self.j_hi[dof_id]
                )

        # Store for debugging
        self._last_base_action = base_action
        self._last_residual_action = residual_action

        # Send combined action using existing position control
        result = self.send_action(total_action)

        # Add residual info to result
        result["base_action"] = base_action
        result["residual_action"] = residual_action
        result["alpha"] = self.alpha

        return result

    def _obs_dict_to_array(self, obs: dict) -> np.ndarray:
        """
        Convert observation dict to array for base policy.

        This should match the format expected by the base policy.
        Customize based on your base policy's input format.
        """
        # Example format matching SO101ResidualEnv observation:
        # [joint_pos(6), joint_vel(6), ee_pos(3)]
        obs_array = []

        # Joint positions
        for joint_name in self.JOINT_NAMES:
            obs_array.append(obs[f"{joint_name}.pos"])

        # Joint velocities
        for joint_name in self.JOINT_NAMES:
            obs_array.append(obs[f"{joint_name}.vel"])

        # End-effector position
        obs_array.extend([obs["ee.pos_x"], obs["ee.pos_y"], obs["ee.pos_z"]])

        return np.array(obs_array, dtype=np.float32)

    def set_residual_mode(self, enabled: bool, base_policy: Optional[Callable] = None):
        """
        Enable or disable residual mode.

        Args:
            enabled: Whether to enable residual mode
            base_policy: Optional new base policy to use
        """
        self.residual_mode = enabled
        if base_policy is not None:
            self.base_policy = base_policy

        logger.info(f"Residual mode {'enabled' if enabled else 'disabled'}")

    def set_residual_params(
        self,
        alpha: Optional[float] = None,
        residual_scale: Optional[float] = None,
    ):
        """
        Update residual parameters.

        Args:
            alpha: New residual blending factor
            residual_scale: New residual scaling factor
        """
        if alpha is not None:
            self.alpha = np.clip(alpha, 0.0, 1.0)
            logger.info(f"Residual alpha set to {self.alpha}")

        if residual_scale is not None:
            self.residual_scale = max(0.0, residual_scale)
            logger.info(f"Residual scale set to {self.residual_scale}")


def create_residual_robot(config, base_policy=None, **residual_kwargs):
    """
    Factory function to create SO101 robot with residual support.

    Args:
        config: SO101MujocoConfig instance
        base_policy: Base policy for residual learning
        **residual_kwargs: Additional arguments for residual control

    Returns:
        SO101 robot instance with residual action support
    """
    from .robot_so101_mujoco import SO101MujocoRobot

    # Create custom class with residual mixin
    class SO101ResidualRobot(ResidualActionMixin, SO101MujocoRobot):
        def __init__(self, config):
            SO101MujocoRobot.__init__(self, config)
            ResidualActionMixin.__init__(
                self,
                base_policy=base_policy,
                **residual_kwargs
            )

        def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
            """Override to support residual mode."""
            if self.residual_mode and "residual" in str(type(action)):
                # If in residual mode and action looks like residual
                return self.send_action_residual(action)
            else:
                # Use normal action sending
                return super().send_action(action)

    return SO101ResidualRobot(config)


# Example usage
if __name__ == "__main__":
    from pathlib import Path

    from lerobot.envs.so101_base_policy import JacobianIKPolicy
    from lerobot.robots.so101_mujoco.configuration_so101_mujoco import SO101MujocoConfig

    # Create config
    config = SO101MujocoConfig()

    # Create base policy
    base_policy = JacobianIKPolicy(max_delta=0.02)

    # Create robot with residual support
    robot = create_residual_robot(
        config,
        base_policy=base_policy,
        alpha=0.5,
        residual_scale=0.02,
        residual_mode=True
    )

    # Connect
    robot.connect()

    # Get observation
    obs = robot.get_observation()
    print(f"Robot connected. EE at: ({obs['ee.pos_x']:.3f}, {obs['ee.pos_y']:.3f}, {obs['ee.pos_z']:.3f})")

    # Test residual action
    residual_action = {
        f"{joint}.pos": 0.1 for joint in robot.JOINT_NAMES
    }

    result = robot.send_action_residual(residual_action)
    print(f"Residual action executed. Alpha: {result['alpha']}")

    robot.disconnect()
    print("Test complete!")