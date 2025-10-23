#!/usr/bin/env python3
"""
Show camera views with position/orientation information.

Creates a composite image showing both camera views with overlaid information
about camera positions and orientations.
"""

import argparse

import cv2
import mujoco as mj
import numpy as np


def create_camera_view_composite(
    world_xml: str = "src/lerobot/envs/so101_assets/paper_square_realistic.xml",
    output_path: str = "camera_views.png",
    image_size: tuple = (640, 480),
):
    """Create composite image showing all camera views with annotations."""

    print("=" * 80)
    print("Camera Views Visualization")
    print("=" * 80)
    print(f"World: {world_xml}")
    print(f"Output: {output_path}")
    print()

    # Load model
    model = mj.MjModel.from_xml_path(world_xml)
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    # Get cameras
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

    print(f"Found {len(cameras)} cameras:")
    for cam in cameras:
        print(f"\n{cam['name']}:")
        print(f"  Position (XYZ): [{cam['position'][0]:.3f}, {cam['position'][1]:.3f}, {cam['position'][2]:.3f}] m")
        print(f"  Forward dir:    [{cam['forward'][0]:.3f}, {cam['forward'][1]:.3f}, {cam['forward'][2]:.3f}]")

    print()
    print("-" * 80)
    print("Rendering camera views...")

    # Render from each camera
    rendered_views = []
    renderer = mj.Renderer(model, image_size[1], image_size[0])

    for cam in cameras:
        renderer.update_scene(data, camera=cam['id'])
        rgb = renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Add text overlay
        y_pos = 30
        cv2.putText(bgr, f"Camera: {cam['name']}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        y_pos += 30
        pos_text = f"Pos: [{cam['position'][0]:.2f}, {cam['position'][1]:.2f}, {cam['position'][2]:.2f}]m"
        cv2.putText(bgr, pos_text, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y_pos += 25
        fwd_text = f"Fwd: [{cam['forward'][0]:.2f}, {cam['forward'][1]:.2f}, {cam['forward'][2]:.2f}]"
        cv2.putText(bgr, fwd_text, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        rendered_views.append(bgr)

    renderer.close()

    # Create composite
    if len(rendered_views) == 2:
        # Side-by-side
        composite = np.hstack(rendered_views)

        # Add title
        title_height = 60
        title_bar = np.zeros((title_height, composite.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_bar, "SO-101 Camera Positions & Views", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        composite = np.vstack([title_bar, composite])

    elif len(rendered_views) == 1:
        composite = rendered_views[0]
    else:
        # Stack vertically if more than 2
        composite = np.vstack(rendered_views)

    # Save
    cv2.imwrite(output_path, composite)

    print(f"✅ Composite image saved: {output_path}")
    print(f"   Size: {composite.shape[1]}x{composite.shape[0]}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Show camera views")
    parser.add_argument(
        "--world",
        type=str,
        default="src/lerobot/envs/so101_assets/paper_square_realistic.xml",
        help="Path to world XML",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="camera_views.png",
        help="Output image path",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Width per camera view",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Height per camera view",
    )

    args = parser.parse_args()

    create_camera_view_composite(
        world_xml=args.world,
        output_path=args.output,
        image_size=(args.width, args.height),
    )


if __name__ == "__main__":
    main()
