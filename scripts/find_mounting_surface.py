#!/usr/bin/env python3
"""
Find the flat mounting surface with four screw holes and dodecagon lens hole.
The camera should be at the center of this surface (center of dodecagon hole).
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
    print("Finding Flat Mounting Surface with Screw Holes")
    print("="*80)

    stl_file = "src/lerobot/envs/so101_assets/official_model/assets/Wrist_Cam_Mount_32x32_UVC_Module_SO101_MIRRORED.stl"

    print(f"\nReading STL file: {stl_file}")
    normals, vertices = read_stl_binary(stl_file)

    # STL is in millimeters, convert to meters
    vertices = vertices * 0.001

    print(f"  Triangles: {len(vertices)}")

    # The mounting surface is a FLAT area, so all triangles should have the same normal
    # and their vertices should be roughly coplanar at the same Y position

    # Looking for triangles in the extension arm region (Y < -0.07)
    # with normals pointing in +Y direction (outward from gripper)

    print("\n" + "="*80)
    print("Analyzing for Flat Mounting Surface")
    print("="*80)

    # Find triangles with normals pointing in +Y direction
    y_facing_triangles = []
    y_facing_normals = []

    for i, (tri_verts, normal) in enumerate(zip(vertices, normals)):
        # Normalize normal
        norm = normal / np.linalg.norm(normal)

        # Check if normal points primarily in +Y direction
        if norm[1] > 0.9:  # Strong +Y component (nearly perpendicular to Y axis)
            # Check if triangle is in extension arm region
            tri_y_values = tri_verts[:, 1]
            if tri_y_values.max() < -0.07:  # In extension arm
                y_facing_triangles.append(tri_verts)
                y_facing_normals.append(norm)

    print(f"\nFound {len(y_facing_triangles)} triangles with strong +Y normals in extension arm")

    if len(y_facing_triangles) > 0:
        y_facing_triangles = np.array(y_facing_triangles)
        y_facing_normals = np.array(y_facing_normals)

        # These should all be on the same flat plane (mounting surface)
        # Get all vertices
        all_verts = y_facing_triangles.reshape(-1, 3)

        print(f"\nMounting surface vertex distribution:")
        print(f"  X range: [{all_verts[:, 0].min():.4f}, {all_verts[:, 0].max():.4f}] m")
        print(f"  Y range: [{all_verts[:, 1].min():.4f}, {all_verts[:, 1].max():.4f}] m")
        print(f"  Z range: [{all_verts[:, 2].min():.4f}, {all_verts[:, 2].max():.4f}] m")

        # The center of the mounting surface (center of dodecagon lens hole)
        center_x = (all_verts[:, 0].min() + all_verts[:, 0].max()) / 2
        center_y = all_verts[:, 1].max()  # At the surface (maximum Y)
        center_z = (all_verts[:, 2].min() + all_verts[:, 2].max()) / 2

        print(f"\nCenter of mounting surface (dodecagon lens hole) in STL frame:")
        print(f"  Position: [{center_x:.4f}, {center_y:.4f}, {center_z:.4f}] m")

        # Average normal of mounting surface
        avg_normal = np.mean(y_facing_normals, axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)

        print(f"  Surface normal: [{avg_normal[0]:.4f}, {avg_normal[1]:.4f}, {avg_normal[2]:.4f}]")

        # Transform to gripper frame
        # Camera mount mesh is at pos="0 -0.000218214 0.000949706" with quat="0 1 0 0"
        # quat="0 1 0 0" means 180° rotation around Y axis
        # Rotation matrix: [[1, 0, 0], [0, -1, 0], [0, 0, -1]]

        mount_pos = np.array([0.0, -0.000218214, 0.000949706])

        # Transform: (x,y,z) -> (x, -y, -z) then add mount_pos
        camera_stl = np.array([center_x, center_y, center_z])
        camera_rotated = np.array([camera_stl[0], -camera_stl[1], -camera_stl[2]])
        camera_gripper = mount_pos + camera_rotated

        # Surface normal after rotation
        normal_rotated = np.array([avg_normal[0], -avg_normal[1], -avg_normal[2]])

        print("\n" + "="*80)
        print("Camera Position in Gripper Frame")
        print("="*80)

        print(f"\nCamera at center of dodecagon lens hole:")
        print(f"  pos=\"{camera_gripper[0]:.4f} {camera_gripper[1]:.4f} {camera_gripper[2]:.4f}\"")

        print(f"\nSurface normal (outward from mount): {normal_rotated}")
        print(f"Camera should point inward (opposite): {-normal_rotated}")

        # The surface normal should be purely +Y (perpendicular to mounting surface)
        print(f"\nNote: If surface normal is close to [0, 1, 0], the mounting surface is truly flat")
        print(f"      and perpendicular to Y axis")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
