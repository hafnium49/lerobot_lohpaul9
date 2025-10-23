#!/usr/bin/env python3
"""
Render camera positions with visual markers showing camera poses.

This script creates an image showing the robot world with visual indicators
for camera positions and view directions.
"""

import argparse

import cv2
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

            cameras.append({
                'name': cam_name,
                'id': i,
                'position': cam_pos,
                'rotation': cam_rot,
                'forward': forward,
            })

    return cameras


def render_scene_with_camera_markers(
    world_xml: str,
    output_path: str,
    view_camera: str = "top_view",
    image_size: tuple = (1280, 960),
    arrow_length: float = 0.15,
):
    """
    Render scene with camera position markers.

    Args:
        world_xml: Path to world XML
        output_path: Where to save output image
        view_camera: Which camera to use for rendering the scene
        image_size: Output image size (width, height)
        arrow_length: Length of direction arrows in meters
    """

    print("=" * 80)
    print("Camera Pose Visualization")
    print("=" * 80)
    print(f"World: {world_xml}")
    print(f"Output: {output_path}")
    print(f"View camera: {view_camera}")
    print(f"Image size: {image_size[0]}x{image_size[1]}")
    print()

    # Load model
    model = mj.MjModel.from_xml_path(world_xml)
    data = mj.MjData(model)

    # Initial simulation step
    mj.mj_forward(model, data)

    # Get camera info
    cameras = get_camera_info(model, data)

    print("Cameras in scene:")
    for cam in cameras:
        print(f"\n{cam['name']}:")
        print(f"  Position: [{cam['position'][0]:.3f}, {cam['position'][1]:.3f}, {cam['position'][2]:.3f}]")
        print(f"  Forward:  [{cam['forward'][0]:.3f}, {cam['forward'][1]:.3f}, {cam['forward'][2]:.3f}]")

    print()
    print("-" * 80)
    print("Rendering scene with camera markers...")

    # Create renderer
    renderer = mj.Renderer(model, image_size[1], image_size[0])

    # Set camera
    view_cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, view_camera)

    # Create scene with camera markers
    scene = mj.MjvScene(model, maxgeom=1000)
    scene_option = mj.MjvOption()

    # Update scene
    mj.mjv_updateScene(
        model, data, scene_option, None, view_cam_id,
        mj.mjtCatBit.mjCAT_ALL, scene
    )

    # Add camera markers to scene
    colors = {
        'top_view': [1.0, 0.0, 0.0, 0.8],       # Red
        'wrist_camera': [0.0, 1.0, 0.0, 0.8],    # Green
    }
    default_color = [0.0, 0.0, 1.0, 0.8]  # Blue

    for cam in cameras:
        color = colors.get(cam['name'], default_color)

        # Add sphere at camera position
        if scene.ngeom < scene.maxgeom:
            mj.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mj.mjtGeom.mjGEOM_SPHERE,
                size=[0.03, 0, 0],  # 3cm radius
                pos=cam['position'],
                mat=np.eye(3).flatten(),
                rgba=color,
            )
            scene.ngeom += 1

        # Add arrow showing view direction
        arrow_end = cam['position'] + cam['forward'] * arrow_length
        arrow_mid = (cam['position'] + arrow_end) / 2

        # Calculate rotation for arrow
        forward = cam['forward']
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, forward)
        rotation_axis_norm = np.linalg.norm(rotation_axis)

        if rotation_axis_norm > 1e-6:
            rotation_axis = rotation_axis / rotation_axis_norm
            cos_angle = np.dot(z_axis, forward)
            angle = np.arccos(np.clip(cos_angle, -1, 1))

            # Rodrigues' rotation
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            rot_mat = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        else:
            rot_mat = np.eye(3) if np.dot(z_axis, forward) > 0 else np.diag([1, 1, -1])

        # Add cylinder (arrow shaft)
        if scene.ngeom < scene.maxgeom:
            mj.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mj.mjtGeom.mjGEOM_CYLINDER,
                size=[0.005, arrow_length / 2, 0],
                pos=arrow_mid,
                mat=rot_mat.flatten(),
                rgba=color,
            )
            scene.ngeom += 1

        # Add cone/cylinder (arrow head)
        cone_size = 0.02
        cone_pos = arrow_end - forward * cone_size * 0.5

        if scene.ngeom < scene.maxgeom:
            mj.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mj.mjtGeom.mjGEOM_CYLINDER,
                size=[0.01, cone_size / 2, 0],
                pos=cone_pos,
                mat=rot_mat.flatten(),
                rgba=color,
            )
            scene.ngeom += 1

    # Render
    renderer.update_scene(data, camera=view_cam_id, scene=scene)
    rgb_array = renderer.render()

    # Convert to BGR for OpenCV
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    # Add text overlay
    text_y = 40
    cv2.putText(bgr_array, "Camera Positions & Orientations", (20, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    text_y += 50
    cv2.putText(bgr_array, "Red sphere + arrow: top_view camera", (20, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    text_y += 35
    cv2.putText(bgr_array, "Green sphere + arrow: wrist_camera (on robot)", (20, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save image
    cv2.imwrite(output_path, bgr_array)

    print(f"✅ Image saved: {output_path}")
    print("=" * 80)

    renderer.close()


def main():
    parser = argparse.ArgumentParser(description="Render camera pose diagram")
    parser.add_argument(
        "--world",
        type=str,
        default="src/lerobot/envs/so101_assets/paper_square_realistic.xml",
        help="Path to world XML",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="camera_poses.png",
        help="Output image path",
    )
    parser.add_argument(
        "--view-camera",
        type=str,
        default="top_view",
        help="Camera to use for rendering the scene",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Image width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=960,
        help="Image height",
    )

    args = parser.parse_args()

    render_scene_with_camera_markers(
        world_xml=args.world,
        output_path=args.output,
        view_camera=args.view_camera,
        image_size=(args.width, args.height),
    )


if __name__ == "__main__":
    main()
