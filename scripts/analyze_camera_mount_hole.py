#!/usr/bin/env python3
"""
Analyze the camera mount STL to find the camera mounting hole position and surface normal.
"""

import numpy as np


def read_stl_binary(filename):
    """Read binary STL file."""
    with open(filename, 'rb') as f:
        # Skip header
        f.read(80)

        # Read number of triangles
        num_triangles = np.frombuffer(f.read(4), dtype=np.uint32)[0]

        vertices = []
        normals = []

        for _ in range(num_triangles):
            # Read normal
            normal = np.frombuffer(f.read(12), dtype=np.float32)
            normals.append(normal)

            # Read 3 vertices
            triangle_vertices = []
            for _ in range(3):
                vertex = np.frombuffer(f.read(12), dtype=np.float32)
                triangle_vertices.append(vertex)
            vertices.append(triangle_vertices)

            # Skip attribute byte count
            f.read(2)

        return np.array(normals), np.array(vertices)


def main():
    print("="*80)
    print("Camera Mount Hole Analysis")
    print("="*80)

    stl_file = "src/lerobot/envs/so101_assets/official_model/assets/Wrist_Cam_Mount_32x32_UVC_Module_SO101_MIRRORED.stl"

    print(f"\nReading STL file: {stl_file}")
    normals, vertices = read_stl_binary(stl_file)

    # STL is in millimeters, convert to meters
    vertices = vertices * 0.001

    print(f"  Triangles: {len(vertices)}")

    # Overall bounds
    all_vertices = vertices.reshape(-1, 3)
    print(f"\nOverall bounds (meters):")
    print(f"  X: [{all_vertices[:, 0].min():.4f}, {all_vertices[:, 0].max():.4f}]")
    print(f"  Y: [{all_vertices[:, 1].min():.4f}, {all_vertices[:, 1].max():.4f}]")
    print(f"  Z: [{all_vertices[:, 2].min():.4f}, {all_vertices[:, 2].max():.4f}]")

    # The camera mount is designed for a 32x32mm UVC camera module
    # The mounting hole should be near the end of the extension arm
    # Looking for features in the extension arm region (large negative Y values)

    # Extension arm is at Y around -0.089 (minimum Y)
    # Camera hole should be at this Y position
    # Looking for the mounting surface normal

    print("\n" + "="*80)
    print("Analyzing Extension Arm Region (where camera mounts)")
    print("="*80)

    # Find vertices near the extension arm tip
    y_threshold = -0.080  # Near the arm extension
    arm_vertices = []
    arm_normals = []

    for i, tri_verts in enumerate(vertices):
        # Check if triangle is in extension arm region
        tri_y_min = tri_verts[:, 1].min()
        tri_y_max = tri_verts[:, 1].max()

        if tri_y_min < y_threshold:
            arm_vertices.append(tri_verts)
            arm_normals.append(normals[i])

    arm_vertices = np.array(arm_vertices)
    arm_normals = np.array(arm_normals)

    print(f"\nFound {len(arm_vertices)} triangles in extension arm region (Y < {y_threshold})")

    if len(arm_vertices) > 0:
        arm_all_verts = arm_vertices.reshape(-1, 3)
        print(f"Extension arm bounds:")
        print(f"  X: [{arm_all_verts[:, 0].min():.4f}, {arm_all_verts[:, 0].max():.4f}]")
        print(f"  Y: [{arm_all_verts[:, 1].min():.4f}, {arm_all_verts[:, 1].max():.4f}]")
        print(f"  Z: [{arm_all_verts[:, 2].min():.4f}, {arm_all_verts[:, 2].max():.4f}]")

        # Camera mounting surface should be facing outward (positive Y direction after mirroring)
        # Look for triangles with normals pointing in +Y direction
        print("\n" + "="*80)
        print("Analyzing Surface Normals")
        print("="*80)

        # Group normals by direction
        normal_directions = {}
        for i, normal in enumerate(arm_normals):
            # Normalize
            norm = normal / np.linalg.norm(normal)

            # Categorize by dominant direction
            dominant_axis = np.argmax(np.abs(norm))
            dominant_sign = np.sign(norm[dominant_axis])
            axis_names = ['X', 'Y', 'Z']
            sign_names = ['-', '+']

            key = f"{sign_names[int((dominant_sign + 1) / 2)]}{axis_names[dominant_axis]}"

            if key not in normal_directions:
                normal_directions[key] = []
            normal_directions[key].append((i, norm))

        print("\nNormal distribution in extension arm:")
        for direction, normals_list in sorted(normal_directions.items()):
            print(f"  {direction}: {len(normals_list)} triangles")

        # Camera mounting surface should face +Y (outward from gripper after rotation)
        if '+Y' in normal_directions:
            print(f"\n" + "="*80)
            print(f"Analyzing +Y facing surface (camera mounting surface)")
            print(f"="*80)

            y_facing_indices = [idx for idx, _ in normal_directions['+Y']]
            y_facing_verts = arm_vertices[y_facing_indices].reshape(-1, 3)

            print(f"Found {len(y_facing_indices)} triangles facing +Y")
            print(f"Mounting surface bounds:")
            print(f"  X: [{y_facing_verts[:, 0].min():.4f}, {y_facing_verts[:, 0].max():.4f}]")
            print(f"  Y: [{y_facing_verts[:, 1].min():.4f}, {y_facing_verts[:, 1].max():.4f}]")
            print(f"  Z: [{y_facing_verts[:, 2].min():.4f}, {y_facing_verts[:, 2].max():.4f}]")

            # Camera hole should be at the center of this surface
            center_x = (y_facing_verts[:, 0].min() + y_facing_verts[:, 0].max()) / 2
            center_y = y_facing_verts[:, 1].max()  # At the surface
            center_z = (y_facing_verts[:, 2].min() + y_facing_verts[:, 2].max()) / 2

            print(f"\nEstimated camera hole center (STL frame):")
            print(f"  Position: [{center_x:.4f}, {center_y:.4f}, {center_z:.4f}] m")

            # Average normal of mounting surface
            avg_normal = np.mean([norm for _, norm in normal_directions['+Y']], axis=0)
            avg_normal = avg_normal / np.linalg.norm(avg_normal)

            print(f"  Surface normal: [{avg_normal[0]:.4f}, {avg_normal[1]:.4f}, {avg_normal[2]:.4f}]")

            print("\n" + "="*80)
            print("Camera Position Recommendation")
            print("="*80)

            # In gripper frame, accounting for camera mount mesh position and rotation
            # Camera mount mesh is at pos="0 -0.000218214 0.000949706" with quat="0 1 0 0"
            # quat="0 1 0 0" means 180° rotation around Y axis
            # This rotation flips X: X_gripper = -X_stl, Y_gripper = Y_stl, Z_gripper = -Z_stl

            mount_pos = np.array([0.0, -0.000218214, 0.000949706])

            # Transform STL coordinates to gripper frame
            # 180° rotation around Y: (x,y,z) -> (-x, y, -z)
            camera_stl = np.array([center_x, center_y, center_z])
            camera_rotated = np.array([-camera_stl[0], camera_stl[1], -camera_stl[2]])
            camera_gripper = mount_pos + camera_rotated

            # Surface normal after rotation
            normal_rotated = np.array([-avg_normal[0], avg_normal[1], -avg_normal[2]])

            print(f"\nCamera position in gripper frame:")
            print(f"  pos=\"{camera_gripper[0]:.4f} {camera_gripper[1]:.4f} {camera_gripper[2]:.4f}\"")

            print(f"\nCamera direction (surface normal after rotation):")
            print(f"  Forward: [{normal_rotated[0]:.4f}, {normal_rotated[1]:.4f}, {normal_rotated[2]:.4f}]")

            # Convert to euler angles
            # Camera looks along -Z in its local frame, we want -Z to align with normal_rotated
            # If normal is [nx, ny, nz], we need pitch and yaw
            # Pitch (rotation around X): angle from horizontal
            pitch = np.arctan2(-normal_rotated[2], np.sqrt(normal_rotated[0]**2 + normal_rotated[1]**2))
            # Yaw (rotation around Z): angle in XY plane
            yaw = np.arctan2(normal_rotated[0], normal_rotated[1])

            print(f"\nRecommended euler angles:")
            print(f"  euler=\"{yaw:.4f} {pitch:.4f} 0\"")
            print(f"  (yaw={np.degrees(yaw):.1f}°, pitch={np.degrees(pitch):.1f}°, roll=0°)")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
