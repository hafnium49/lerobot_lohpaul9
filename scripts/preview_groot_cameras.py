#!/usr/bin/env python3
"""
Quickly render and save the base-policy camera views for GR00T.

This is useful to verify the dual-camera inputs (e.g., top_view + wrist_camera)
without running a full rollout or loading the GR00T model.

Usage:
  PYTHONPATH=src python scripts/preview_groot_cameras.py \
    --cameras top_view wrist_camera \
    --image-size 480 640 \
    --output-dir /tmp/groot_views
"""

import argparse
from pathlib import Path

import cv2

import importlib.util
import sys
from pathlib import Path


def _load_so101_residual_env():
    """
    Load SO101ResidualEnv directly from file to avoid heavy package imports.
    This sidesteps optional deps (e.g., deepdiff) when we only need rendering.
    """
    env_path = Path(__file__).parent / "src" / "lerobot" / "envs" / "so101_residual_env.py"
    if not env_path.exists():
        # Fall back to repo-relative path if running from repo root
        env_path = Path(__file__).parent.parent / "src" / "lerobot" / "envs" / "so101_residual_env.py"

    if not env_path.exists():
        raise FileNotFoundError(f"Cannot find so101_residual_env.py (looked in {env_path})")

    spec = importlib.util.spec_from_file_location("so101_residual_env", env_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["so101_residual_env"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.SO101ResidualEnv


def main():
    parser = argparse.ArgumentParser(description="Preview GR00T base-policy camera views.")
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=["top_view", "wrist_camera"],
        help="Camera names to render (provide two for dual view).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=[480, 640],
        help="Image size (height width) for renders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/groot_views",
        help="Directory to save preview PNGs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for env reset.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create env with image observations and dual cameras for the base policy.
    SO101ResidualEnv = _load_so101_residual_env()

    env = SO101ResidualEnv(
        use_image_obs=True,
        image_size=tuple(args.image_size),
        camera_name_for_obs=args.cameras[0],  # first camera used for obs if needed
        base_camera_names=tuple(args.cameras),
        seed=args.seed,
    )

    env.reset()
    renders = env._render_base_cameras()

    # Normalize to list
    if not isinstance(renders, list):
        renders = [renders]

    for idx, (cam_name, img_rgb) in enumerate(zip(args.cameras, renders)):
        # Save as BGR for OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        outfile = out_dir / f"groot_view_{idx}_{cam_name}.png"
        cv2.imwrite(str(outfile), img_bgr)
        print(f"Saved {cam_name} -> {outfile}")


if __name__ == "__main__":
    main()
