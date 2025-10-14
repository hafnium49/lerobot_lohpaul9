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
Base policy implementations for SO101 residual RL.

This module provides base policies that the RL agent learns to correct:
1. JacobianIKPolicy: Uses Jacobian-based IK for end-effector control
2. FrozenILPolicy: Uses a frozen pre-trained IL model (e.g., ACT, Diffusion)
3. ZeroPolicy: Returns zero actions (pure RL baseline)
"""

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch


class BasePolicy(ABC):
    """Abstract base class for base policies."""

    @abstractmethod
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute base action given observation.

        Args:
            obs: Environment observation

        Returns:
            action: Joint position deltas (shape: [6])
        """
        pass

    def reset(self):
        """Reset any internal state (optional)."""
        pass


class JacobianIKPolicy(BasePolicy):
    """
    Jacobian-based inverse kinematics policy.

    This policy uses the robot's Jacobian to move the end-effector
    toward the paper to push it into the target square.
    """

    def __init__(
        self,
        kp_xyz: float = 0.5,
        kp_ori: float = 0.3,
        max_delta: float = 0.02,
        lambda_dls: float = 0.01,
    ):
        """
        Initialize Jacobian IK policy.

        Args:
            kp_xyz: Position control gain
            kp_ori: Orientation control gain
            max_delta: Maximum joint delta per step
            lambda_dls: Damped least squares regularization
        """
        self.kp_xyz = kp_xyz
        self.kp_ori = kp_ori
        self.max_delta = max_delta
        self.lambda_dls = lambda_dls

        # Robot kinematics parameters (simplified SO-101)
        self.link_lengths = np.array([0.16, 0.212, 0.212, 0.12])  # Approximate

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute IK action to push paper toward target.

        The observation contains:
        - Joint positions [6]
        - Joint velocities [6]
        - Paper pose [7] (xyz + quat)
        - Goal vector [3] (paper to tape)
        - EE position [3]
        """
        # Parse observation
        joint_pos = obs[:6]
        paper_pos = obs[12:15]  # Paper x, y, z
        goal_vec = obs[19:22]  # Vector from paper to tape
        ee_pos = obs[22:25]  # Current EE position

        # Compute desired EE motion
        # Strategy: Move EE behind paper to push it toward goal
        paper_to_goal = goal_vec[:2]  # Only x, y
        push_direction = paper_to_goal / (np.linalg.norm(paper_to_goal) + 1e-6)

        # Position EE slightly behind paper in push direction
        offset = 0.08  # Distance behind paper
        desired_ee_xy = paper_pos[:2] - push_direction * offset
        desired_ee_z = max(paper_pos[2] + 0.02, 0.05)  # Slightly above paper

        desired_ee = np.array([desired_ee_xy[0], desired_ee_xy[1], desired_ee_z])

        # Compute error
        ee_error = desired_ee - ee_pos
        ee_error *= self.kp_xyz

        # Compute approximate Jacobian (3-link for XYZ control)
        J = self._compute_jacobian_3dof(joint_pos[:3])

        # Damped least squares IK
        JJT = J @ J.T + self.lambda_dls * np.eye(3)
        joint_delta_xyz = J.T @ np.linalg.solve(JJT, ee_error)

        # Full joint delta (6 DOF)
        joint_delta = np.zeros(6)
        joint_delta[:3] = joint_delta_xyz

        # Simple wrist control: keep gripper pointed down
        wrist_target = -np.pi/2  # Point down
        joint_delta[3] = self.kp_ori * (wrist_target - joint_pos[3])

        # Keep wrist roll neutral
        joint_delta[4] = -self.kp_ori * joint_pos[4]

        # Gripper: open when far from paper, close when near
        dist_to_paper = np.linalg.norm(ee_pos[:2] - paper_pos[:2])
        if dist_to_paper < 0.1:
            joint_delta[5] = -0.01  # Close gripper
        else:
            joint_delta[5] = 0.01  # Open gripper

        # Clip to maximum delta
        joint_delta = np.clip(joint_delta, -self.max_delta, self.max_delta)

        return joint_delta.astype(np.float32)

    def _compute_jacobian_3dof(self, q: np.ndarray) -> np.ndarray:
        """
        Compute approximate 3-DOF Jacobian for first 3 joints.

        This is a simplified version for demonstration.
        In practice, use the actual robot kinematics.
        """
        # Simplified kinematic Jacobian (3x3)
        # This would normally come from forward kinematics
        c1, s1 = np.cos(q[0]), np.sin(q[0])
        c2, s2 = np.cos(q[1]), np.sin(q[1])
        c23 = np.cos(q[1] + q[2])
        s23 = np.sin(q[1] + q[2])

        L1, L2, L3 = self.link_lengths[:3]

        # Position Jacobian (simplified)
        J = np.array([
            [-s1*(L2*c2 + L3*c23), c1*(L2*s2 + L3*s23), c1*L3*s23],
            [ c1*(L2*c2 + L3*c23), s1*(L2*s2 + L3*s23), s1*L3*s23],
            [0, L2*c2 + L3*c23, L3*c23]
        ])

        return J


class FrozenILPolicy(BasePolicy):
    """
    Frozen imitation learning policy.

    This loads a pre-trained policy (e.g., ACT, Diffusion Policy)
    and uses it as the base policy for residual learning.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        policy_type: str = "act",
        device: str = "auto",
        action_scale: float = 0.02,
    ):
        """
        Initialize frozen IL policy.

        Args:
            checkpoint_path: Path to model checkpoint
            policy_type: Type of policy ("act", "diffusion", etc.)
            device: Device for inference ("cpu", "cuda", "auto")
            action_scale: Scaling factor for actions
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.policy_type = policy_type
        self.action_scale = action_scale

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load policy
        self.policy = self._load_policy()

        # History buffer for policies that need temporal context
        self.obs_history = []
        self.history_len = 1  # Adjust based on policy needs

    def _load_policy(self):
        """Load the pre-trained policy model."""
        try:
            # This is a placeholder - actual implementation would load
            # from LeRobot's policy factory
            from lerobot.policies import make_policy

            # Load checkpoint
            policy = make_policy(
                self.checkpoint_path,
                device=self.device
            )
            policy.eval()  # Set to evaluation mode
            return policy

        except Exception as e:
            warnings.warn(f"Failed to load IL policy: {e}. Using zero policy as fallback.")
            return None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Get action from frozen IL policy.

        Args:
            obs: Environment observation

        Returns:
            action: Joint position deltas
        """
        if self.policy is None:
            return np.zeros(6, dtype=np.float32)

        try:
            # Add to history
            self.obs_history.append(obs)
            if len(self.obs_history) > self.history_len:
                self.obs_history.pop(0)

            # Convert to torch tensor
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

            # Get action from policy
            with torch.no_grad():
                if self.policy_type == "act":
                    # ACT expects observation dict
                    obs_dict = {
                        "joint_pos": obs_tensor[:, :6],
                        "joint_vel": obs_tensor[:, 6:12],
                        # Add other expected inputs based on policy
                    }
                    action = self.policy.predict(obs_dict)
                else:
                    # Generic policy interface
                    action = self.policy.select_action(obs_tensor)

            # Convert to numpy
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy().squeeze()

            # Ensure correct shape
            if action.shape != (6,):
                action = action[:6] if len(action) >= 6 else np.zeros(6)

            # Scale action
            action = action * self.action_scale

            return action.astype(np.float32)

        except Exception as e:
            warnings.warn(f"IL policy inference failed: {e}")
            return np.zeros(6, dtype=np.float32)

    def reset(self):
        """Reset observation history."""
        self.obs_history = []


class ZeroPolicy(BasePolicy):
    """
    Zero policy that returns no action.

    This is used as a baseline for pure RL without any base policy.
    """

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Return zero action."""
        return np.zeros(6, dtype=np.float32)


class HybridPolicy(BasePolicy):
    """
    Hybrid policy that combines multiple base policies.

    This can switch between or blend different base policies
    based on the task phase or performance.
    """

    def __init__(
        self,
        policies: list[BasePolicy],
        weights: Optional[np.ndarray] = None,
        mode: str = "blend",  # "blend" or "switch"
    ):
        """
        Initialize hybrid policy.

        Args:
            policies: List of base policies
            weights: Blending weights for each policy
            mode: How to combine policies ("blend" or "switch")
        """
        self.policies = policies
        self.mode = mode

        if weights is None:
            weights = np.ones(len(policies)) / len(policies)
        self.weights = np.array(weights)

        self.active_idx = 0  # For switching mode

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute hybrid action.

        Args:
            obs: Environment observation

        Returns:
            action: Combined action from policies
        """
        if self.mode == "blend":
            # Weighted average of all policies
            action = np.zeros(6, dtype=np.float32)
            for policy, weight in zip(self.policies, self.weights):
                action += weight * policy(obs)
            return action

        elif self.mode == "switch":
            # Use single active policy
            return self.policies[self.active_idx](obs)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def switch_policy(self, idx: int):
        """Switch to a different policy (for switch mode)."""
        if 0 <= idx < len(self.policies):
            self.active_idx = idx

    def reset(self):
        """Reset all policies."""
        for policy in self.policies:
            policy.reset()


# Test base policies
if __name__ == "__main__":
    print("Testing Base Policies...")

    # Create dummy observation
    obs = np.random.randn(25).astype(np.float32)

    # Test Jacobian IK policy
    print("\n1. Testing Jacobian IK Policy:")
    ik_policy = JacobianIKPolicy()
    action = ik_policy(obs)
    print(f"   Action shape: {action.shape}")
    print(f"   Action range: [{action.min():.3f}, {action.max():.3f}]")

    # Test Zero policy
    print("\n2. Testing Zero Policy:")
    zero_policy = ZeroPolicy()
    action = zero_policy(obs)
    print(f"   Action: {action}")

    # Test Hybrid policy
    print("\n3. Testing Hybrid Policy:")
    hybrid = HybridPolicy([ik_policy, zero_policy], weights=[0.7, 0.3])
    action = hybrid(obs)
    print(f"   Blended action shape: {action.shape}")
    print(f"   Blended action range: [{action.min():.3f}, {action.max():.3f}]")

    print("\nAll base policies working!")