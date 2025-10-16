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
Gymnasium environment wrapper for SO101 MuJoCo robot with residual RL support.

This environment enables residual reinforcement learning on top of a base policy
(e.g., frozen IL model or Jacobian IK controller) for the paper-in-square task.
"""

import warnings
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import mujoco as mj
import numpy as np
from gymnasium.spaces import Box, Dict


class SO101ResidualEnv(gym.Env):
    """
    Gymnasium environment for SO101 robot with residual RL.

    The environment implements a paper-sliding task where the robot must
    slide a piece of paper into a red tape square on the table.

    Key features:
    - Residual action space: RL policy adds corrections to base policy
    - Multi-rate control: 360Hz physics, 180Hz control, 30Hz policy
    - Domain randomization for sim-to-real transfer
    - Dense reward shaping with success bonus
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        xml_path: str = None,
        base_policy=None,
        alpha: float = 0.5,
        act_scale: float = 0.02,
        residual_penalty: float = 0.001,
        frame_skip: int = 12,  # 360Hz / 30Hz = 12
        randomize: bool = True,
        render_mode: str = None,
        camera_name: str = "top",
        seed: int = None,
    ):
        """
        Initialize SO101 residual RL environment.

        Args:
            xml_path: Path to MuJoCo XML file
            base_policy: Base policy that provides nominal actions (callable)
            alpha: Residual blending factor (0=base only, 1=full residual)
            act_scale: Scaling factor for residual actions
            residual_penalty: L2 penalty coefficient for residual magnitude
            frame_skip: Number of physics steps per policy step
            randomize: Enable domain randomization
            render_mode: Rendering mode ('human' or 'rgb_array')
            camera_name: Camera to use for rendering
            seed: Random seed
        """
        super().__init__()

        # Default XML path if not provided
        if xml_path is None:
            xml_path = Path(__file__).parent / "so101_assets" / "paper_square_realistic.xml"

        # Load MuJoCo model
        self.model = mj.MjModel.from_xml_path(str(xml_path))
        self.data = mj.MjData(self.model)

        # Control parameters
        self.base_policy = base_policy
        self.alpha = alpha
        self.act_scale = act_scale
        self.residual_penalty = residual_penalty
        self.frame_skip = frame_skip
        self.randomize = randomize

        # Rendering
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.viewer = None

        # Get joint and body IDs
        self._setup_ids()

        # Define action and observation spaces
        self.n_joints = 6  # 5 arm joints + gripper

        # Action space: residual joint deltas
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_joints,),
            dtype=np.float32
        )

        # Observation space: joint pos/vel + paper pose + goal vector + EE pos
        obs_dim = (
            self.n_joints * 2 +  # Joint positions and velocities
            7 +  # Paper pose (x, y, z, quat)
            3 +  # Goal vector (paper center to tape center)
            3    # End-effector position
        )
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Task parameters (updated for centered robot base at floor origin)
        self.tape_center = np.array([0.275, 0.175])  # From XML (translated)
        # A5 target frame: 210mm x 148mm (rotated 90°: long edge along X-axis)
        self.tape_half_size = np.array([0.105, 0.074])  # Half-dimensions (X, Y)
        self.paper_half_size = np.array([0.074, 0.105])  # A5 paper (X, Y)

        # Episode tracking
        self.steps = 0
        self.max_steps = 400  # ~13 seconds at 30Hz

        # Gripper fingertip IDs for runtime friction control (release hack)
        self.fixed_fingertip_id = None
        self.moving_fingertip_id = None
        self.gripper_base_friction = 1.0  # Normal grasping friction
        self.gripper_release_friction = 0.6  # Lowered during opening
        self.gripper_release_threshold = 0.5  # Gripper position threshold for release mode

        # Try to get fingertip geom IDs
        try:
            self.fixed_fingertip_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "fixed_fingertip")
            self.moving_fingertip_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "moving_fingertip")
            if self.fixed_fingertip_id >= 0 and self.moving_fingertip_id >= 0:
                print("✅ Gripper fingertips found - release control enabled")
        except:
            pass  # Fingertips not available, no problem

        # Set random seed if provided
        if seed is not None:
            self.seed(seed)

    def _setup_ids(self):
        """Get MuJoCo IDs for joints, bodies, and sites."""
        # Joint IDs (official SO-101 uses "gripper" for single coupled gripper joint)
        self.joint_names = [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper"
        ]
        self.joint_ids = [
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ]

        # Body IDs
        self.paper_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "paper")
        self.tape_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "tape_square")
        self.ee_site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "ee_site")

        # Paper corner site IDs for success detection
        self.corner_site_ids = [
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, f"paper_corner{i}")
            for i in range(1, 5)
        ]

    def _get_obs(self) -> np.ndarray:
        """Get current observation vector."""
        # Joint positions and velocities
        qpos = np.array([self.data.qpos[jid] for jid in self.joint_ids])
        qvel = np.array([self.data.qvel[jid] for jid in self.joint_ids])

        # Paper pose (position + quaternion)
        paper_pos = self.data.xpos[self.paper_body_id].copy()
        paper_quat = self.data.xquat[self.paper_body_id].copy()
        paper_pose = np.concatenate([paper_pos, paper_quat])

        # Goal vector (paper center to tape center)
        goal_vec = np.zeros(3)
        goal_vec[:2] = self.tape_center - paper_pos[:2]

        # End-effector position
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()

        # Concatenate all observations
        obs = np.concatenate([qpos, qvel, paper_pose, goal_vec, ee_pos])
        return obs.astype(np.float32)

    def _get_paper_corners_world(self) -> np.ndarray:
        """Get paper corner positions in world frame."""
        corners = np.zeros((4, 3))
        for i, site_id in enumerate(self.corner_site_ids):
            corners[i] = self.data.site_xpos[site_id].copy()
        return corners

    def _check_success(self) -> bool:
        """Check if all paper corners are inside the tape frame (rectangular)."""
        corners = self._get_paper_corners_world()

        # Check if all corners are within rectangular tape frame bounds (2D check)
        # tape_half_size is [half_x, half_y] for the rectangular A5 frame
        tape_min = self.tape_center - self.tape_half_size
        tape_max = self.tape_center + self.tape_half_size

        for corner in corners:
            if not (tape_min[0] <= corner[0] <= tape_max[0] and
                    tape_min[1] <= corner[1] <= tape_max[1]):
                return False
        return True

    def _update_gripper_friction(self):
        """
        Modulate fingertip friction based on gripper state (release hack).

        When gripper is opening (releasing paper), temporarily lower
        friction to prevent "glued paper" syndrome. This simulates the
        natural release behavior of nitrile gloves.
        """
        if self.fixed_fingertip_id is None or self.moving_fingertip_id is None:
            return  # Fingertips not available
        if self.fixed_fingertip_id < 0 or self.moving_fingertip_id < 0:
            return  # Invalid IDs

        # Get current gripper position (joint 5)
        gripper_pos = self.data.qpos[self.joint_ids[5]]

        # If gripper is opening (pos > threshold), use release friction
        if gripper_pos > self.gripper_release_threshold:
            target_friction = self.gripper_release_friction
        else:
            target_friction = self.gripper_base_friction

        # Update friction coefficients (only slide component, keep spin/roll)
        self.model.geom_friction[self.fixed_fingertip_id, 0] = target_friction
        self.model.geom_friction[self.moving_fingertip_id, 0] = target_friction

    def _compute_reward(
        self,
        action: np.ndarray,
        success: bool,
        paper_pos: np.ndarray
    ) -> tuple[float, dict]:
        """
        Compute reward for current state and action.

        Returns:
            reward: Scalar reward
            info: Dictionary with reward components
        """
        # Distance from paper center to tape center
        dist_to_goal = np.linalg.norm(self.tape_center - paper_pos[:2])

        # Orientation penalty (prefer aligned paper)
        paper_quat = self.data.xquat[self.paper_body_id]
        # Convert quaternion to euler angles
        paper_euler = self._quat_to_euler(paper_quat)
        orientation_penalty = abs(paper_euler[2])  # Yaw angle

        # Residual magnitude penalty
        residual_penalty = np.sum(action**2)

        # Compute reward components
        reward = 0.0
        reward_info = {}

        # Success bonus
        if success:
            reward += 10.0
            reward_info["success_bonus"] = 10.0

        # Distance shaping (negative reward, closer is better)
        dist_reward = -2.0 * dist_to_goal
        reward += dist_reward
        reward_info["dist_reward"] = dist_reward

        # Orientation shaping
        ori_reward = -0.1 * orientation_penalty
        reward += ori_reward
        reward_info["ori_reward"] = ori_reward

        # Residual penalty
        res_penalty = -self.residual_penalty * residual_penalty
        reward += res_penalty
        reward_info["residual_penalty"] = res_penalty

        # Time penalty (small, encourages efficiency)
        time_penalty = -0.01
        reward += time_penalty
        reward_info["time_penalty"] = time_penalty

        reward_info["total_reward"] = reward
        reward_info["dist_to_goal"] = dist_to_goal

        return reward, reward_info

    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        w, x, y, z = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset MuJoCo simulation
        mj.mj_resetData(self.model, self.data)

        # Set robot to home position
        home_pos = np.array([0.0, 0.3, -0.6, 0.0, 0.0, 0.0])
        for i, jid in enumerate(self.joint_ids):
            self.data.qpos[jid] = home_pos[i]

        # Randomize initial paper position if enabled
        if self.randomize and hasattr(self, "np_random"):
            # Random paper position
            paper_x = self.np_random.uniform(0.25, 0.35)
            paper_y = self.np_random.uniform(-0.1, 0.1)
            paper_z = 0.001

            # Random paper orientation (small)
            paper_yaw = self.np_random.uniform(-0.3, 0.3)

            # Get paper joint ID (free joint has 7 DOF: 3 pos + 4 quat)
            paper_joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "paper_free")
            paper_qpos_addr = self.model.jnt_qposadr[paper_joint_id]

            # Set paper position
            self.data.qpos[paper_qpos_addr:paper_qpos_addr+3] = [paper_x, paper_y, paper_z]

            # Set paper orientation (quaternion from yaw)
            quat = np.array([np.cos(paper_yaw/2), 0, 0, np.sin(paper_yaw/2)])
            self.data.qpos[paper_qpos_addr+3:paper_qpos_addr+7] = quat

            # Randomize friction slightly
            if self.randomize:
                # Find paper geom
                paper_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "paper_geom")
                # Vary friction by ±20%
                base_friction = np.array([0.60, 0.002, 0.0001])
                friction_scale = self.np_random.uniform(0.8, 1.2)
                self.model.geom_friction[paper_geom_id] = base_friction * friction_scale

        # Randomize gripper friction (simulates dust/humidity on nitrile gloves)
        if self.randomize and self.fixed_fingertip_id is not None:
            if self.fixed_fingertip_id >= 0 and self.moving_fingertip_id >= 0:
                # Vary base/release friction by ±15%
                friction_scale = self.np_random.uniform(0.85, 1.15)
                self.gripper_base_friction = 1.0 * friction_scale
                self.gripper_release_friction = 0.6 * friction_scale

        # Forward dynamics to update derived quantities
        mj.mj_forward(self.model, self.data)

        # Reset episode counters
        self.steps = 0

        # Get initial observation
        obs = self._get_obs()
        info = {"paper_pos": self.data.xpos[self.paper_body_id].copy()}

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one environment step with residual action.

        Args:
            action: Residual action from RL policy (scaled -1 to 1)

        Returns:
            obs: New observation
            reward: Reward for this step
            terminated: Whether episode ended (success)
            truncated: Whether episode was cut short (time limit)
            info: Additional information
        """
        # Clip and scale residual action
        action = np.clip(action, -1.0, 1.0)
        residual_action = action * self.act_scale

        # Get base action from base policy if available
        base_action = np.zeros(self.n_joints)
        if self.base_policy is not None:
            try:
                obs = self._get_obs()
                base_action = self.base_policy(obs)
                base_action = np.asarray(base_action, dtype=np.float32)
                if base_action.shape != (self.n_joints,):
                    base_action = np.zeros(self.n_joints)
            except Exception as e:
                warnings.warn(f"Base policy failed: {e}")
                base_action = np.zeros(self.n_joints)

        # Combine base and residual actions
        total_action = base_action + self.alpha * residual_action

        # Apply action as joint position targets
        for i, jid in enumerate(self.joint_ids):
            # Current position + delta
            target_pos = self.data.qpos[jid] + total_action[i]
            # Clip to joint limits
            if jid < len(self.model.jnt_range):
                joint_range = self.model.jnt_range[jid]
                target_pos = np.clip(target_pos, joint_range[0], joint_range[1])
            # Set control
            self.data.ctrl[i] = target_pos

        # Note: Official SO-101 model has coupled gripper fingers (handled by single actuator)
        # No need to mirror right finger as in simplified model

        # Update gripper friction based on state (release hack)
        self._update_gripper_friction()

        # Step physics
        for _ in range(self.frame_skip):
            mj.mj_step(self.model, self.data)

        # Get new observation
        obs = self._get_obs()

        # Check success
        success = self._check_success()

        # Compute reward
        paper_pos = self.data.xpos[self.paper_body_id].copy()
        reward, reward_info = self._compute_reward(action, success, paper_pos)

        # Check termination conditions
        self.steps += 1
        terminated = success  # Episode ends on success
        truncated = self.steps >= self.max_steps  # Time limit

        # Prepare info dict
        info = {
            "success": success,
            "steps": self.steps,
            "paper_pos": paper_pos,
            "base_action": base_action,
            "residual_action": residual_action,
            "total_action": total_action,
            **reward_info
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            # Offscreen rendering
            camera_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, self.camera_name)
            if camera_id == -1:
                camera_id = None  # Use default camera

            # Render to RGB array
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            depth = np.zeros((480, 640), dtype=np.float32)
            mj.mj_renderFrame(self.model, self.data, rgb, depth)
            return rgb

        elif self.render_mode == "human":
            # Interactive viewer (requires mujoco-python-viewer)
            try:
                import mujoco_viewer
                if self.viewer is None:
                    self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
                self.viewer.render()
            except ImportError:
                warnings.warn("mujoco-python-viewer not installed, cannot render in human mode")
            return None

    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed: int):
        """Set random seed."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


# Test the environment
if __name__ == "__main__":
    print("Testing SO101 Residual Environment...")

    # Create environment
    env = SO101ResidualEnv(randomize=True)

    # Test reset
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial paper position: {info['paper_pos']}")

    # Test random actions
    total_reward = 0
    for i in range(100):
        # Random residual action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if i % 20 == 0:
            print(f"Step {i}: reward={reward:.3f}, success={info['success']}, "
                  f"dist={info['dist_to_goal']:.3f}")

        if terminated or truncated:
            print(f"Episode ended at step {i+1}: success={info['success']}")
            break

    print(f"Total reward: {total_reward:.2f}")
    env.close()