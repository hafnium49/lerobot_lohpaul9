#!/usr/bin/env python3
"""
Visualize camera positions with visible markers in MuJoCo viewer.

This script modifies the world to add visual markers (spheres and arrows)
showing camera positions and orientations.

Usage:
    python scripts/visualize_cameras_with_markers.py
"""

import argparse
import tempfile
import time
from pathlib import Path

import mujoco as mj
import numpy as np
from lxml import etree


def get_camera_info(model, data):
    """Get information about all cameras."""
    cameras = []
    for i in range(model.ncam):
        cam_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i)
        if cam_name:
            cam_pos = data.cam_xpos[i].copy()
            cam_rot = data.cam_xmat[i].reshape(3, 3).copy()
            forward = -cam_rot[:, 2]
            cameras.append({
                'name': cam_name,
                'id': i,
                'position': cam_pos,
                'forward': forward,
            })
    return cameras


def add_camera_markers_to_xml(
    world_xml: str,
    arrow_length: float = 0.15,
) -> str:
    """
    Add visual markers for cameras to the XML.

    Returns:
        Path to temporary XML file with markers added
    """

    # Parse the XML
    tree = etree.parse(world_xml)
    root = tree.getroot()

    # Find or create worldbody
    worldbody = root.find('worldbody')
    if worldbody is None:
        worldbody = etree.SubElement(root, 'worldbody')

    # Load model temporarily to get camera positions at initial state
    temp_model = mj.MjModel.from_xml_path(world_xml)
    temp_data = mj.MjData(temp_model)
    mj.mj_forward(temp_model, temp_data)
    cameras = get_camera_info(temp_model, temp_data)

    # Define colors for cameras
    colors = {
        'top_view': '1 0 0 0.8',       # Red
        'wrist_camera': '0 1 0 0.8',   # Green
    }
    default_color = '0 0 1 0.8'  # Blue

    # Add markers for each camera
    for cam in cameras:
        color = colors.get(cam['name'], default_color)

        # Create a body for camera markers at origin (we'll use absolute positions)
        marker_body = etree.SubElement(worldbody, 'body', name=f'{cam["name"]}_marker')
        marker_body.set('pos', '0 0 0')  # Body at origin, use absolute positions in geoms

        # Add sphere at camera position
        etree.SubElement(marker_body, 'geom', {
            'name': f'{cam["name"]}_sphere',
            'type': 'sphere',
            'pos': f'{cam["position"][0]} {cam["position"][1]} {cam["position"][2]}',
            'size': '0.03',
            'rgba': color,
            'contype': '0',
            'conaffinity': '0',
        })

        # Add arrow showing direction
        arrow_end = cam['position'] + cam['forward'] * arrow_length

        # Add cylinder (arrow shaft) using fromto
        etree.SubElement(marker_body, 'geom', {
            'name': f'{cam["name"]}_arrow_shaft',
            'type': 'cylinder',
            'fromto': f'{cam["position"][0]} {cam["position"][1]} {cam["position"][2]} '
                     f'{arrow_end[0]} {arrow_end[1]} {arrow_end[2]}',
            'size': '0.005',
            'rgba': color,
            'contype': '0',
            'conaffinity': '0',
        })

        # Add cone/cylinder (arrow head)
        forward = cam['forward']
        cone_start = arrow_end - forward * 0.02
        etree.SubElement(marker_body, 'geom', {
            'name': f'{cam["name"]}_arrow_head',
            'type': 'cylinder',
            'fromto': f'{cone_start[0]} {cone_start[1]} {cone_start[2]} '
                     f'{arrow_end[0]} {arrow_end[1]} {arrow_end[2]}',
            'size': '0.01',
            'rgba': color,
            'contype': '0',
            'conaffinity': '0',
        })

    # Save to temporary file in the same directory as the original
    # (to preserve relative paths for mesh files)
    original_path = Path(world_xml)
    temp_path = original_path.parent / f'_temp_camera_markers_{original_path.name}'

    with open(temp_path, 'wb') as f:
        tree.write(f, pretty_print=True, xml_declaration=True, encoding='utf-8')

    return str(temp_path)


def visualize_cameras_with_markers(
    world_xml: str = "src/lerobot/envs/so101_assets/paper_square_realistic.xml",
    duration: int = 60,
):
    """
    Launch MuJoCo viewer with camera position markers visible.

    Args:
        world_xml: Path to original world XML
        duration: How long to run (seconds, 0 = infinite)
    """

    print("=" * 80)
    print("Camera Visualization with Markers")
    print("=" * 80)
    print(f"Original world: {world_xml}")
    print()

    # Create modified XML with camera markers
    print("Adding camera markers to world...")
    modified_xml = add_camera_markers_to_xml(world_xml)
    print(f"✅ Created temporary world with markers: {modified_xml}")
    print()

    # Load the modified model
    model = mj.MjModel.from_xml_path(modified_xml)
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    # Get camera info
    cameras = get_camera_info(model, data)

    # Print camera information
    print("=" * 80)
    print(f"Found {len(cameras)} camera(s)")
    print("=" * 80)
    for i, cam in enumerate(cameras, 1):
        print(f"\nCamera {i}: {cam['name']}")
        print(f"  Position: [{cam['position'][0]:.3f}, {cam['position'][1]:.3f}, {cam['position'][2]:.3f}] m")
        print(f"  Forward:  [{cam['forward'][0]:.3f}, {cam['forward'][1]:.3f}, {cam['forward'][2]:.3f}]")

    print("\n" + "=" * 80)
    print("Launching MuJoCo Viewer with Camera Markers")
    print("=" * 80)
    print("\nVISUAL MARKERS:")
    print("  🔴 Red sphere + arrow: top_view camera")
    print("  🟢 Green sphere + arrow: wrist_camera")
    print("  Arrow shows the direction the camera is pointing")
    print()
    print("CONTROLS:")
    print("  - Left mouse drag: Rotate view")
    print("  - Right mouse drag: Pan view")
    print("  - Scroll wheel: Zoom")
    print("  - TAB: Cycle through camera views")
    print("  - '[' / ']': Previous/next camera")
    print("  - ESC or close window: Exit")
    print()
    print(f"Viewer will run for {duration} seconds (or until closed)")
    print("=" * 80)
    print()

    try:
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
                    print("Visualization time expired. Closing viewer...")
                    break

                time.sleep(0.01)

        print("\n✅ Viewer closed successfully")

    except Exception as e:
        print(f"\n❌ Could not launch viewer: {e}")
        print("This is normal in headless environments.")

    finally:
        # Clean up temporary file
        import os
        try:
            os.unlink(modified_xml)
            print(f"🗑️  Cleaned up temporary file")
        except:
            pass

    print("\n" + "=" * 80)
    print("Visualization complete")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Visualize cameras with visible markers")
    parser.add_argument(
        "--world",
        type=str,
        default="src/lerobot/envs/so101_assets/paper_square_realistic.xml",
        help="Path to world XML",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration in seconds (0 = infinite)",
    )

    args = parser.parse_args()

    visualize_cameras_with_markers(
        world_xml=args.world,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()
