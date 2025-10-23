#!/usr/bin/env python3
"""
Visualize camera positions and directions in MuJoCo viewer.

This script shows the SO-101 world in the MuJoCo viewer with visual markers
indicating camera positions and orientations (view directions).

Usage:
    python scripts/visualize_camera_poses.py
"""

import argparse
import time

import mujoco as mj
import mujoco.viewer as viewer
import numpy as np


def get_camera_pose(model, data, camera_name):
    """
    Get camera position and orientation.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        camera_name: Name of camera

    Returns:
        pos: Camera position (3,)
        rot_mat: Camera rotation matrix (3, 3)
        forward: Forward direction vector (3,)
    """
    cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, camera_name)

    # Get camera position
    cam_pos = data.cam_xpos[cam_id].copy()

    # Get camera rotation matrix
    cam_rot = data.cam_xmat[cam_id].reshape(3, 3).copy()

    # Camera forward direction is -Z axis in camera frame
    # (MuJoCo cameras look down -Z axis)
    forward = -cam_rot[:, 2]

    return cam_pos, cam_rot, forward


def add_camera_visualization_geoms(model, data, camera_name, arrow_length=0.15, arrow_radius=0.005):
    """
    Add visualization geoms for a camera (this is conceptual - we'll use viewer overlays instead).

    Returns:
        Camera position, rotation, and forward vector for visualization.
    """
    cam_pos, cam_rot, forward = get_camera_pose(model, data, camera_name)

    # Calculate arrow endpoint
    arrow_end = cam_pos + forward * arrow_length

    return {
        'name': camera_name,
        'position': cam_pos,
        'rotation': cam_rot,
        'forward': forward,
        'arrow_end': arrow_end,
        'arrow_length': arrow_length,
        'arrow_radius': arrow_radius,
    }


def visualize_cameras(
    world_xml: str = "src/lerobot/envs/so101_assets/paper_square_realistic.xml",
    cameras_to_show: list = None,
    duration: int = 60,
):
    """
    Visualize camera positions and orientations in MuJoCo viewer.

    Args:
        world_xml: Path to MuJoCo XML file
        cameras_to_show: List of camera names to visualize (None = all)
        duration: How long to run visualization (seconds, 0 = infinite)
    """

    # Load model
    print("=" * 80)
    print("Camera Pose Visualization")
    print("=" * 80)
    print(f"Loading world: {world_xml}")

    model = mj.MjModel.from_xml_path(world_xml)
    data = mj.MjData(model)

    # Get list of all cameras
    all_cameras = []
    for i in range(model.ncam):
        cam_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i)
        if cam_name:
            all_cameras.append(cam_name)

    if cameras_to_show is None:
        cameras_to_show = all_cameras

    print(f"\nAvailable cameras: {all_cameras}")
    print(f"Visualizing: {cameras_to_show}\n")

    # Initial simulation step
    mj.mj_forward(model, data)

    # Get camera visualization data
    camera_viz_data = {}
    for cam_name in cameras_to_show:
        if cam_name in all_cameras:
            camera_viz_data[cam_name] = add_camera_visualization_geoms(
                model, data, cam_name, arrow_length=0.15, arrow_radius=0.005
            )
            cam_data = camera_viz_data[cam_name]
            print(f"Camera: {cam_name}")
            print(f"  Position: [{cam_data['position'][0]:.3f}, {cam_data['position'][1]:.3f}, {cam_data['position'][2]:.3f}]")
            print(f"  Forward:  [{cam_data['forward'][0]:.3f}, {cam_data['forward'][1]:.3f}, {cam_data['forward'][2]:.3f}]")
            print()

    print("=" * 80)
    print("Opening MuJoCo Viewer...")
    print()
    print("CONTROLS:")
    print("  - Left mouse: Rotate view")
    print("  - Right mouse: Move view")
    print("  - Scroll: Zoom")
    print("  - Double-click: Select object")
    print("  - Ctrl+Right mouse: Move light")
    print()
    print("Camera visualization:")
    print("  - Colored spheres: Camera positions")
    print("  - Arrows: Camera view directions")
    print("  - Red: top_view camera")
    print("  - Green: wrist_camera camera")
    print()
    print(f"Viewer will run for {duration} seconds (or until you close the window)")
    print("=" * 80)

    # Create viewer with scene options callback
    def add_visual_markers(scn):
        """Add visual markers for cameras to the scene."""

        # Define colors for each camera
        colors = {
            'top_view': [1.0, 0.0, 0.0, 1.0],      # Red
            'wrist_camera': [0.0, 1.0, 0.0, 1.0],   # Green
        }
        default_color = [0.0, 0.0, 1.0, 1.0]  # Blue for others

        for cam_name, cam_data in camera_viz_data.items():
            color = colors.get(cam_name, default_color)

            # Add sphere at camera position
            mj.mjv_initGeom(
                scn.geoms[scn.ngeom],
                type=mj.mjtGeom.mjGEOM_SPHERE,
                size=[0.02, 0, 0],  # Radius = 2cm
                pos=cam_data['position'],
                mat=np.eye(3).flatten(),
                rgba=color,
            )
            scn.ngeom += 1

            # Add arrow showing view direction
            # Arrow is represented as a cylinder
            arrow_start = cam_data['position']
            arrow_end = cam_data['arrow_end']
            arrow_midpoint = (arrow_start + arrow_end) / 2

            # Calculate rotation to align cylinder with arrow direction
            forward = cam_data['forward']
            # Default cylinder is along Z axis, rotate to align with forward
            z_axis = np.array([0, 0, 1])

            # Rotation axis
            rotation_axis = np.cross(z_axis, forward)
            rotation_axis_norm = np.linalg.norm(rotation_axis)

            if rotation_axis_norm > 1e-6:
                rotation_axis = rotation_axis / rotation_axis_norm
                # Rotation angle
                cos_angle = np.dot(z_axis, forward)
                angle = np.arccos(np.clip(cos_angle, -1, 1))

                # Rodrigues' rotation formula
                K = np.array([
                    [0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0]
                ])
                rot_mat = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            else:
                # Already aligned or opposite
                if np.dot(z_axis, forward) > 0:
                    rot_mat = np.eye(3)
                else:
                    rot_mat = np.diag([1, 1, -1])

            # Add cylinder (arrow shaft)
            mj.mjv_initGeom(
                scn.geoms[scn.ngeom],
                type=mj.mjtGeom.mjGEOM_CYLINDER,
                size=[cam_data['arrow_radius'], cam_data['arrow_length'] / 2, 0],
                pos=arrow_midpoint,
                mat=rot_mat.flatten(),
                rgba=color,
            )
            scn.ngeom += 1

            # Add cone (arrow head)
            cone_size = 0.015  # 1.5cm
            cone_pos = arrow_end - forward * cone_size * 0.5

            mj.mjv_initGeom(
                scn.geoms[scn.ngeom],
                type=mj.mjtGeom.mjGEOM_CYLINDER,  # Cone would be better but cylinder works
                size=[cam_data['arrow_radius'] * 2, cone_size / 2, 0],
                pos=cone_pos,
                mat=rot_mat.flatten(),
                rgba=color,
            )
            scn.ngeom += 1

    # Launch viewer with custom rendering
    with mj.viewer.launch_passive(model, data) as v:
        start_time = time.time()

        while v.is_running():
            # Step simulation
            mj.mj_step(model, data)

            # Update camera visualization data (in case robot moves)
            for cam_name in cameras_to_show:
                if cam_name in all_cameras:
                    camera_viz_data[cam_name] = add_camera_visualization_geoms(
                        model, data, cam_name, arrow_length=0.15, arrow_radius=0.005
                    )

            # Add visual markers to scene
            with v.lock():
                v.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = True
                add_visual_markers(v.scn)

            # Sync viewer
            v.sync()

            # Check duration
            if duration > 0 and (time.time() - start_time) > duration:
                print("\nVisualization time expired. Closing viewer...")
                break

            # Small sleep to prevent high CPU usage
            time.sleep(0.01)

    print("\nViewer closed.")


def main():
    parser = argparse.ArgumentParser(description="Visualize camera poses in MuJoCo viewer")
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
        help="Duration to run visualization in seconds (0 = infinite)",
    )

    args = parser.parse_args()

    visualize_cameras(
        world_xml=args.world,
        cameras_to_show=args.cameras,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()
