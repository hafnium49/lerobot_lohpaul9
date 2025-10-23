#!/usr/bin/env python3
"""
Visualize camera positions and directions in MuJoCo.

This script shows camera information and optionally launches the viewer
if a display is available.

Usage:
    python scripts/visualize_camera_poses_fixed.py --no-viewer  # Info only
    python scripts/visualize_camera_poses_fixed.py              # With viewer
"""

import argparse
import time

import mujoco as mj
import numpy as np


def get_camera_info(model, data):
    """Get information about all cameras in the scene."""
    cameras = []

    for i in range(model.ncam):
        cam_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i)
        if cam_name:
            # Get camera pose
            cam_pos = data.cam_xpos[i].copy()
            cam_rot = data.cam_xmat[i].reshape(3, 3).copy()
            forward = -cam_rot[:, 2]  # Camera looks down -Z axis

            # Get other useful info
            up = cam_rot[:, 1]  # Up direction
            right = cam_rot[:, 0]  # Right direction

            cameras.append({
                'name': cam_name,
                'id': i,
                'position': cam_pos,
                'rotation': cam_rot,
                'forward': forward,
                'up': up,
                'right': right,
            })

    return cameras


def print_camera_info(cameras):
    """Print detailed camera information."""
    print("\n" + "=" * 80)
    print(f"Found {len(cameras)} camera(s) in the scene")
    print("=" * 80)

    for i, cam in enumerate(cameras, 1):
        print(f"\nCamera {i}: {cam['name']}")
        print("-" * 80)
        print(f"Position (XYZ):    [{cam['position'][0]:7.3f}, {cam['position'][1]:7.3f}, {cam['position'][2]:7.3f}] m")
        print(f"Forward direction: [{cam['forward'][0]:7.3f}, {cam['forward'][1]:7.3f}, {cam['forward'][2]:7.3f}]")
        print(f"Up direction:      [{cam['up'][0]:7.3f}, {cam['up'][1]:7.3f}, {cam['up'][2]:7.3f}]")
        print(f"Right direction:   [{cam['right'][0]:7.3f}, {cam['right'][1]:7.3f}, {cam['right'][2]:7.3f}]")

        # Interpret position
        print(f"\nInterpretation:")
        print(f"  Height above ground: {cam['position'][2]:.3f} m ({cam['position'][2]*100:.1f} cm)")

        # Interpret forward direction
        if abs(cam['forward'][2] + 1.0) < 0.01:
            print(f"  Looking: Straight down (bird's eye view)")
        elif abs(cam['forward'][2]) < 0.1:
            print(f"  Looking: Horizontally")
        else:
            angle_deg = np.arcsin(-cam['forward'][2]) * 180 / np.pi
            print(f"  Looking: {angle_deg:.1f}° {'down' if cam['forward'][2] < 0 else 'up'} from horizontal")

        # Horizontal direction
        if abs(cam['forward'][0]) > 0.1 or abs(cam['forward'][1]) > 0.1:
            horiz_angle = np.arctan2(cam['forward'][1], cam['forward'][0]) * 180 / np.pi
            print(f"  Horizontal angle: {horiz_angle:.1f}° (0°=+X, 90°=+Y)")

    print("\n" + "=" * 80)


def visualize_cameras(
    world_xml: str = "src/lerobot/envs/so101_assets/paper_square_realistic.xml",
    cameras_to_show: list = None,
    duration: int = 60,
    launch_viewer: bool = True,
):
    """
    Visualize camera positions and orientations.

    Args:
        world_xml: Path to MuJoCo XML file
        cameras_to_show: List of camera names to visualize (None = all)
        duration: How long to run visualization (seconds, 0 = infinite)
        launch_viewer: Whether to launch the interactive viewer
    """

    print("=" * 80)
    print("Camera Pose Visualization")
    print("=" * 80)
    print(f"Loading world: {world_xml}")

    # Load model
    model = mj.MjModel.from_xml_path(world_xml)
    data = mj.MjData(model)

    # Initial simulation step
    mj.mj_forward(model, data)

    # Get camera info
    cameras = get_camera_info(model, data)

    # Filter cameras if requested
    if cameras_to_show:
        cameras = [c for c in cameras if c['name'] in cameras_to_show]

    # Print camera information
    print_camera_info(cameras)

    # Launch viewer if requested
    if launch_viewer:
        print("\n" + "=" * 80)
        print("Launching MuJoCo Viewer...")
        print("=" * 80)
        print("\nVIEWER CONTROLS:")
        print("  - Left mouse drag: Rotate view")
        print("  - Right mouse drag: Move view")
        print("  - Scroll wheel: Zoom in/out")
        print("  - Double-click: Select object")
        print("  - TAB: Cycle through cameras")
        print("  - '[': Previous camera")
        print("  - ']': Next camera")
        print("  - ESC or close window: Exit")
        print()
        print(f"Viewer will run for {duration} seconds (or until you close it)")
        print("=" * 80)

        try:
            # Try to launch viewer
            import mujoco.viewer

            with mujoco.viewer.launch_passive(model, data) as viewer:
                start_time = time.time()

                while viewer.is_running():
                    # Step simulation
                    mj.mj_step(model, data)

                    # Sync viewer
                    viewer.sync()

                    # Check duration
                    if duration > 0 and (time.time() - start_time) > duration:
                        print("\nVisualization time expired. Closing viewer...")
                        break

                    # Small sleep
                    time.sleep(0.01)

            print("\nViewer closed successfully.")

        except Exception as e:
            print(f"\n⚠️  Could not launch viewer: {e}")
            print("This is normal in headless environments (no display).")
            print("Camera information has been printed above.")

    else:
        print("\nViewer disabled (--no-viewer flag used)")

    print("\n" + "=" * 80)
    print("Visualization complete")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize camera poses in MuJoCo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show camera info and launch viewer
  python scripts/visualize_camera_poses_fixed.py

  # Show only camera info (no viewer)
  python scripts/visualize_camera_poses_fixed.py --no-viewer

  # Show specific cameras only
  python scripts/visualize_camera_poses_fixed.py --cameras top_view wrist_camera

  # Run viewer for 30 seconds
  python scripts/visualize_camera_poses_fixed.py --duration 30
        """
    )
    parser.add_argument(
        "--world",
        type=str,
        default="src/lerobot/envs/so101_assets/paper_square_realistic.xml",
        help="Path to MuJoCo XML world file",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=None,
        help="Camera names to visualize (default: all cameras)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration to run viewer in seconds (0 = infinite)",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Don't launch viewer, only print camera information",
    )

    args = parser.parse_args()

    visualize_cameras(
        world_xml=args.world,
        cameras_to_show=args.cameras,
        duration=args.duration,
        launch_viewer=not args.no_viewer,
    )


if __name__ == "__main__":
    main()
