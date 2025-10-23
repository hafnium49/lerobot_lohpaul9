#!/usr/bin/env python3
"""
Mirror an STL file across a specified plane.

This script loads an STL file and creates a mirrored version.

Usage:
    python scripts/mirror_stl.py input.stl output.stl --axis X
"""

import argparse

import numpy as np


def read_stl_ascii(filename):
    """Read ASCII STL file."""
    vertices = []
    normals = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    current_vertices = []
    for line in lines:
        line = line.strip()
        if line.startswith('facet normal'):
            parts = line.split()
            normal = [float(parts[2]), float(parts[3]), float(parts[4])]
            normals.append(normal)
        elif line.startswith('vertex'):
            parts = line.split()
            vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
            current_vertices.append(vertex)
            if len(current_vertices) == 3:
                vertices.append(current_vertices)
                current_vertices = []

    return np.array(normals), np.array(vertices)


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


def read_stl(filename):
    """Read STL file (auto-detect ASCII or binary)."""
    with open(filename, 'rb') as f:
        header = f.read(80)

    # Check if ASCII (starts with "solid")
    try:
        if header[:5].decode('ascii').lower() == 'solid':
            return read_stl_ascii(filename)
    except:
        pass

    # Otherwise binary
    return read_stl_binary(filename)


def write_stl_binary(filename, normals, vertices):
    """Write binary STL file."""
    with open(filename, 'wb') as f:
        # Write header (80 bytes)
        header = b'Mirrored STL file' + b' ' * (80 - len(b'Mirrored STL file'))
        f.write(header)

        # Write number of triangles
        num_triangles = len(vertices)
        f.write(np.uint32(num_triangles).tobytes())

        # Write triangles
        for i in range(num_triangles):
            # Write normal
            f.write(normals[i].astype(np.float32).tobytes())

            # Write 3 vertices
            for j in range(3):
                f.write(vertices[i, j].astype(np.float32).tobytes())

            # Write attribute byte count (0)
            f.write(np.uint16(0).tobytes())


def mirror_stl(input_file, output_file, axis='X'):
    """
    Mirror an STL file across a plane.

    Args:
        input_file: Input STL file path
        output_file: Output STL file path
        axis: Mirror axis ('X', 'Y', or 'Z')
    """

    print(f"Reading STL file: {input_file}")
    normals, vertices = read_stl(input_file)

    print(f"  Triangles: {len(vertices)}")
    print(f"  Mirroring across {axis} axis")

    # Create mirror matrix
    mirror_matrix = np.eye(3)
    if axis.upper() == 'X':
        mirror_matrix[0, 0] = -1
        axis_idx = 0
    elif axis.upper() == 'Y':
        mirror_matrix[1, 1] = -1
        axis_idx = 1
    elif axis.upper() == 'Z':
        mirror_matrix[2, 2] = -1
        axis_idx = 2
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be X, Y, or Z")

    # Mirror vertices
    mirrored_vertices = vertices.copy()
    for i in range(len(mirrored_vertices)):
        for j in range(3):
            mirrored_vertices[i, j] = mirror_matrix @ vertices[i, j]

    # Reverse vertex order to maintain correct face orientation
    mirrored_vertices = mirrored_vertices[:, ::-1, :]

    # Mirror normals
    mirrored_normals = normals.copy()
    for i in range(len(mirrored_normals)):
        mirrored_normals[i] = mirror_matrix @ normals[i]

    # Recalculate normals from vertices to ensure correctness
    for i in range(len(mirrored_vertices)):
        v1 = mirrored_vertices[i, 1] - mirrored_vertices[i, 0]
        v2 = mirrored_vertices[i, 2] - mirrored_vertices[i, 0]
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        mirrored_normals[i] = normal

    print(f"Writing mirrored STL file: {output_file}")
    write_stl_binary(output_file, mirrored_normals, mirrored_vertices)

    print("✅ Done!")
    print(f"\nOriginal bounds:")
    print(f"  X: [{vertices[:,:,0].min():.3f}, {vertices[:,:,0].max():.3f}]")
    print(f"  Y: [{vertices[:,:,1].min():.3f}, {vertices[:,:,1].max():.3f}]")
    print(f"  Z: [{vertices[:,:,2].min():.3f}, {vertices[:,:,2].max():.3f}]")

    print(f"\nMirrored bounds:")
    print(f"  X: [{mirrored_vertices[:,:,0].min():.3f}, {mirrored_vertices[:,:,0].max():.3f}]")
    print(f"  Y: [{mirrored_vertices[:,:,1].min():.3f}, {mirrored_vertices[:,:,1].max():.3f}]")
    print(f"  Z: [{mirrored_vertices[:,:,2].min():.3f}, {mirrored_vertices[:,:,2].max():.3f}]")


def main():
    parser = argparse.ArgumentParser(description="Mirror an STL file")
    parser.add_argument("input", help="Input STL file")
    parser.add_argument("output", help="Output STL file")
    parser.add_argument(
        "--axis",
        choices=['X', 'Y', 'Z', 'x', 'y', 'z'],
        default='X',
        help="Mirror axis (default: X)"
    )

    args = parser.parse_args()

    mirror_stl(args.input, args.output, args.axis)


if __name__ == "__main__":
    main()
