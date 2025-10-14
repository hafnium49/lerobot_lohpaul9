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
PPO training script for SO101 residual reinforcement learning.

This script trains a residual RL policy on top of a base policy (Jacobian IK or frozen IL)
for the paper-in-square task using Stable-Baselines3.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

# Import our custom environment and policies
from lerobot.envs.so101_base_policy import (
    BasePolicy,
    HybridPolicy,
    JacobianIKPolicy,
    ZeroPolicy,
)
from lerobot.envs.so101_residual_env import SO101ResidualEnv


class ResidualRLCallback(BaseCallback):
    """
    Custom callback for logging residual RL metrics.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_successes = []
        self.residual_magnitudes = []

    def _on_step(self) -> bool:
        # Log additional metrics from info
        if "success" in self.locals.get("infos", [{}])[0]:
            info = self.locals["infos"][0]
            self.episode_successes.append(info["success"])

            # Log to tensorboard
            if self.logger:
                self.logger.record("residual/success_rate",
                                 np.mean(self.episode_successes[-100:]))
                if "residual_penalty" in info:
                    self.logger.record("residual/penalty", info["residual_penalty"])
                if "dist_to_goal" in info:
                    self.logger.record("task/dist_to_goal", info["dist_to_goal"])

        return True

    def _on_rollout_end(self) -> None:
        # Log rollout statistics
        if self.logger and len(self.episode_successes) > 0:
            self.logger.record("residual/recent_success_rate",
                             np.mean(self.episode_successes[-10:]))


def make_env(
    env_id: str,
    rank: int,
    seed: int,
    base_policy: BasePolicy,
    alpha: float,
    act_scale: float,
    residual_penalty: float,
    randomize: bool,
    xml_path: Optional[Path] = None,
):
    """
    Create a single environment instance.

    Args:
        env_id: Environment identifier
        rank: Rank of the environment (for parallel envs)
        seed: Random seed
        base_policy: Base policy for residual learning
        alpha: Residual blending factor
        act_scale: Action scaling factor
        residual_penalty: L2 penalty for residual magnitude
        randomize: Enable domain randomization
        xml_path: Path to MuJoCo XML file

    Returns:
        Thunk that creates the environment
    """
    def _thunk():
        env = SO101ResidualEnv(
            xml_path=xml_path,
            base_policy=base_policy,
            alpha=alpha,
            act_scale=act_scale,
            residual_penalty=residual_penalty,
            randomize=randomize,
            seed=seed + rank,
        )
        env = Monitor(env)
        return env

    return _thunk


def create_base_policy(policy_type: str, **kwargs) -> BasePolicy:
    """
    Create base policy based on type.

    Args:
        policy_type: Type of base policy ("jacobian", "zero", "il", "hybrid")
        **kwargs: Additional arguments for policy

    Returns:
        Base policy instance
    """
    if policy_type == "jacobian":
        return JacobianIKPolicy(**kwargs)
    elif policy_type == "zero":
        return ZeroPolicy()
    elif policy_type == "il":
        # For IL policy, need checkpoint path
        from lerobot.envs.so101_base_policy import FrozenILPolicy
        return FrozenILPolicy(**kwargs)
    elif policy_type == "hybrid":
        # Create hybrid of Jacobian and Zero
        policies = [
            JacobianIKPolicy(**kwargs.get("ik_kwargs", {})),
            ZeroPolicy(),
        ]
        return HybridPolicy(policies, weights=[0.7, 0.3])
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def train_residual_rl(args):
    """
    Main training function for residual RL.

    Args:
        args: Command-line arguments
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"so101_residual_{args.base_policy}_{timestamp}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting residual RL training: {exp_name}")
    print(f"Output directory: {output_dir}")

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create base policy
    base_policy_kwargs = {}
    if args.base_policy == "jacobian":
        base_policy_kwargs = {
            "kp_xyz": args.jacobian_kp_xyz,
            "kp_ori": args.jacobian_kp_ori,
            "max_delta": args.act_scale,
        }
    elif args.base_policy == "il" and args.il_checkpoint:
        base_policy_kwargs = {
            "checkpoint_path": args.il_checkpoint,
            "device": args.device,
        }

    base_policy = create_base_policy(args.base_policy, **base_policy_kwargs)

    # Create vectorized training environments
    print(f"Creating {args.n_envs} parallel environments...")
    env_kwargs = {
        "base_policy": base_policy,
        "alpha": args.alpha,
        "act_scale": args.act_scale,
        "residual_penalty": args.residual_penalty,
        "randomize": True,
        "xml_path": args.xml_path,
    }

    if args.n_envs == 1:
        # Single environment
        env = DummyVecEnv([make_env("SO101ResidualEnv", 0, args.seed, **env_kwargs)])
    else:
        # Multiple parallel environments
        if args.vec_env_type == "dummy":
            env = DummyVecEnv([
                make_env("SO101ResidualEnv", i, args.seed, **env_kwargs)
                for i in range(args.n_envs)
            ])
        else:
            # Subprocess environments (more efficient but can't share objects)
            # Need to create base policy in each subprocess
            env = SubprocVecEnv([
                make_env("SO101ResidualEnv", i, args.seed, **env_kwargs)
                for i in range(args.n_envs)
            ])

    env = VecMonitor(env, str(output_dir / "monitor"))

    # Create evaluation environment
    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs["randomize"] = args.eval_randomize
    eval_env = DummyVecEnv([
        make_env("SO101ResidualEnv", 0, args.seed + 1000, **eval_env_kwargs)
    ])

    # PPO hyperparameters
    ppo_kwargs = {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "clip_range_vf": args.clip_range_vf,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "device": args.device,
        "tensorboard_log": str(output_dir / "tensorboard"),
        "verbose": 1,
    }

    # Network architecture
    policy_kwargs = {
        "net_arch": {
            "pi": [args.hidden_size] * args.n_layers,  # Policy network
            "vf": [args.hidden_size] * args.n_layers,  # Value network
        },
        "activation_fn": torch.nn.Tanh if args.activation == "tanh" else torch.nn.ReLU,
    }

    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        **ppo_kwargs
    )

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_residual",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)

    # Custom residual RL callback
    residual_callback = ResidualRLCallback(verbose=1)
    callbacks.append(residual_callback)

    callback_list = CallbackList(callbacks)

    # Train the model
    print(f"\nStarting training for {args.total_timesteps} timesteps...")
    print(f"Base policy: {args.base_policy}")
    print(f"Alpha (residual blend): {args.alpha}")
    print(f"Action scale: {args.act_scale}")
    print(f"Residual penalty: {args.residual_penalty}")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback_list,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Save final model
    final_model_path = output_dir / "final_model"
    model.save(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}")

    # Final evaluation
    print("\nRunning final evaluation...")
    from stable_baselines3.common.evaluation import evaluate_policy

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.n_final_eval,
        deterministic=True,
    )
    print(f"Final performance: {mean_reward:.2f} ± {std_reward:.2f}")

    # Save training configuration
    config_path = output_dir / "config.txt"
    with open(config_path, "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    print(f"\nTraining complete! Results saved to: {output_dir}")

    # Clean up
    env.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train residual RL policy for SO101 paper-in-square task"
    )

    # Environment arguments
    parser.add_argument("--xml-path", type=str, default=None,
                      help="Path to MuJoCo XML file")
    parser.add_argument("--n-envs", type=int, default=4,
                      help="Number of parallel environments")
    parser.add_argument("--vec-env-type", type=str, default="dummy",
                      choices=["dummy", "subproc"],
                      help="Type of vectorized environment")

    # Base policy arguments
    parser.add_argument("--base-policy", type=str, default="jacobian",
                      choices=["jacobian", "zero", "il", "hybrid"],
                      help="Type of base policy")
    parser.add_argument("--il-checkpoint", type=str, default=None,
                      help="Path to IL policy checkpoint (if using IL base)")
    parser.add_argument("--jacobian-kp-xyz", type=float, default=0.5,
                      help="Position gain for Jacobian IK")
    parser.add_argument("--jacobian-kp-ori", type=float, default=0.3,
                      help="Orientation gain for Jacobian IK")

    # Residual RL arguments
    parser.add_argument("--alpha", type=float, default=0.5,
                      help="Residual blending factor (0=base only, 1=full residual)")
    parser.add_argument("--act-scale", type=float, default=0.02,
                      help="Action scaling factor")
    parser.add_argument("--residual-penalty", type=float, default=0.001,
                      help="L2 penalty for residual magnitude")

    # PPO arguments
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                      help="Total training timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                      help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=256,
                      help="Number of steps per rollout")
    parser.add_argument("--batch-size", type=int, default=64,
                      help="Batch size for PPO updates")
    parser.add_argument("--n-epochs", type=int, default=10,
                      help="Number of epochs per PPO update")
    parser.add_argument("--gamma", type=float, default=0.99,
                      help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                      help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2,
                      help="PPO clip range")
    parser.add_argument("--clip-range-vf", type=float, default=None,
                      help="Value function clip range")
    parser.add_argument("--ent-coef", type=float, default=0.005,
                      help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                      help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                      help="Maximum gradient norm")

    # Network arguments
    parser.add_argument("--hidden-size", type=int, default=128,
                      help="Hidden layer size")
    parser.add_argument("--n-layers", type=int, default=2,
                      help="Number of hidden layers")
    parser.add_argument("--activation", type=str, default="tanh",
                      choices=["tanh", "relu"],
                      help="Activation function")

    # Training arguments
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                      help="Device (cpu, cuda, auto)")
    parser.add_argument("--output-dir", type=str, default="runs/residual_rl",
                      help="Output directory")

    # Evaluation arguments
    parser.add_argument("--eval-freq", type=int, default=10000,
                      help="Evaluation frequency (timesteps)")
    parser.add_argument("--n-eval-episodes", type=int, default=10,
                      help="Number of evaluation episodes")
    parser.add_argument("--n-final-eval", type=int, default=50,
                      help="Number of final evaluation episodes")
    parser.add_argument("--eval-randomize", action="store_true",
                      help="Use randomization during evaluation")

    # Saving arguments
    parser.add_argument("--save-freq", type=int, default=25000,
                      help="Checkpoint save frequency (timesteps)")

    args = parser.parse_args()

    # Validate arguments
    if args.base_policy == "il" and args.il_checkpoint is None:
        parser.error("--il-checkpoint required when using IL base policy")

    # Run training
    train_residual_rl(args)


if __name__ == "__main__":
    main()