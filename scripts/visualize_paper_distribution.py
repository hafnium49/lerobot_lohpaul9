#!/usr/bin/env python3
"""
Visualize the distribution of paper positions in the SO-101 environment.

This script:
1. Initializes the environment with randomization enabled.
2. Collects N samples of paper poses (position + orientation).
3. Renders a base image from the top_view camera.
4. Projects the paper corners onto the image plane for each sample.
5. Overlays the paper rectangles on the base image to show the distribution.
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from lerobot.envs.so101_residual_env import SO101ResidualEnv
from scipy.spatial.transform import Rotation

def get_camera_matrix(model, data, camera_name, width, height):
    """
    Compute the 3x4 camera projection matrix P.
    Projects world point (x,y,z) to pixel coordinates (u,v).
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    
    # Get camera position and orientation
    pos = data.cam_xpos[cam_id]
    mat = data.cam_xmat[cam_id].reshape(3, 3)
    
    # Extrinsic matrix (World -> Camera)
    # R is the transpose of the camera orientation matrix
    # t is -R * pos
    R_world_cam = mat.T
    t_world_cam = -R_world_cam @ pos
    
    # OpenGL camera looks down -Z, OpenCV looks down +Z
    # We need to rotate 180 deg around X axis to convert
    R_gl_cv = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    
    R = R_gl_cv @ R_world_cam
    t = R_gl_cv @ t_world_cam
    
    # Intrinsic matrix
    fovy = model.cam_fovy[cam_id]
    f = 0.5 * height / np.tan(np.deg2rad(fovy) / 2)
    K = np.array([
        [f, 0, width / 2],
        [0, f, height / 2],
        [0, 0, 1]
    ])
    
    # Projection matrix P = K [R | t]
    P = K @ np.hstack((R, t.reshape(3, 1)))
    
    return P

def project_point(P, point_world):
    """Project a 3D world point to 2D pixel coordinates."""
    point_h = np.append(point_world, 1.0)  # Homogeneous coordinates
    point_img_h = P @ point_h
    u = point_img_h[0] / point_img_h[2]
    v = point_img_h[1] / point_img_h[2]
    return np.array([u, v])

def get_paper_corners(pos, quat, half_size):
    """
    Calculate the 4 corners of the paper in world coordinates.
    
    Args:
        pos: (3,) paper position
        quat: (4,) paper quaternion (w, x, y, z)
        half_size: (2,) paper half-dimensions (x, y)
    """
    # Paper corners in local frame (z=0)
    # A5 paper is rectangular
    dx, dy = half_size
    local_corners = np.array([
        [dx, dy, 0],
        [-dx, dy, 0],
        [-dx, -dy, 0],
        [dx, -dy, 0]
    ])
    
    # Rotate and translate to world frame
    rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses (x, y, z, w)
    world_corners = rot.apply(local_corners) + pos
    
    return world_corners

def main():
    print("Initializing environment...")
    env = SO101ResidualEnv(randomize=True, use_image_obs=False)
    
    # Camera settings
    camera_name = "top_view"
    width, height = 640, 480
    
    # Reset to get initial state and render base image
    print("Rendering base image...")
    env.reset(seed=42)
    
    # Create renderer manually to get the specific camera view
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    renderer.update_scene(env.data, camera=cam_id)
    base_image = renderer.render()
    
    # Get camera projection matrix
    P = get_camera_matrix(env.model, env.data, camera_name, width, height)
    
    # Collect samples
    n_samples = 50
    print(f"Collecting {n_samples} samples...")
    
    paper_corners_list = []
    
    for i in range(n_samples):
        # Reset environment to get a new random paper position
        # We use a different seed each time to ensure randomness
        _, info = env.reset(seed=42 + i)
        
        paper_pos = info["paper_pos"]
        # Get paper quaternion directly from data
        paper_quat = env.data.xquat[env.paper_body_id]
        
        # Calculate corners in world space
        corners_world = get_paper_corners(paper_pos, paper_quat, env.paper_half_size)
        
        # Project to image space
        corners_img = np.array([project_point(P, p) for p in corners_world])
        paper_corners_list.append(corners_img)
        
    env.close()
    
    # Visualization
    print("Generating visualization...")
    fig, ax = plt.subplots(figsize=(10, 7.5))
    
    # Display base image
    ax.imshow(base_image)
    
    # Overlay paper rectangles
    # Use a colormap to distinguish individual samples slightly, or just use a single transparent color
    for i, corners in enumerate(paper_corners_list):
        # Create a polygon
        poly = patches.Polygon(corners, closed=True, 
                               facecolor='cyan', edgecolor='blue', 
                               alpha=0.1, linewidth=1)
        ax.add_patch(poly)
        
    # Add title and labels
    ax.set_title(f'Paper Position Distribution (N={n_samples})\nTop View Camera', 
                 color='white', fontsize=14, weight='bold')
    ax.axis('off')
    
    # Save figure
    output_file = 'paper_distribution.png'
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=150)
    print(f"✅ Saved visualization to {output_file}")

if __name__ == "__main__":
    main()
