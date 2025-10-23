#!/usr/bin/env python3
"""
Visualize an STL file using matplotlib.

Usage:
    python scripts/visualize_stl.py <stl_file>
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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


def visualize_stl(filename, output_file=None):
    """Visualize STL file."""
    print(f"Reading STL file: {filename}")
    normals, vertices = read_stl(filename)

    print(f"  Triangles: {len(vertices)}")
    print(f"  Bounds:")
    print(f"    X: [{vertices[:,:,0].min():.3f}, {vertices[:,:,0].max():.3f}]")
    print(f"    Y: [{vertices[:,:,1].min():.3f}, {vertices[:,:,1].max():.3f}]")
    print(f"    Z: [{vertices[:,:,2].min():.3f}, {vertices[:,:,2].max():.3f}]")

    # Create figure with multiple views
    fig = plt.figure(figsize=(15, 10))

    # Create 3D plot
    ax = fig.add_subplot(2, 2, 1, projection='3d')

    # Create mesh
    mesh = Poly3DCollection(vertices, alpha=0.7, edgecolor='k', linewidths=0.1)
    mesh.set_facecolor([0.5, 0.5, 1.0])
    ax.add_collection3d(mesh)

    # Set equal aspect ratio
    max_range = np.array([
        vertices[:,:,0].max() - vertices[:,:,0].min(),
        vertices[:,:,1].max() - vertices[:,:,1].min(),
        vertices[:,:,2].max() - vertices[:,:,2].min()
    ]).max() / 2.0

    mid_x = (vertices[:,:,0].max() + vertices[:,:,0].min()) * 0.5
    mid_y = (vertices[:,:,1].max() + vertices[:,:,1].min()) * 0.5
    mid_z = (vertices[:,:,2].max() + vertices[:,:,2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D View')

    # Top view (XY plane)
    ax2 = fig.add_subplot(2, 2, 2)
    for tri in vertices[::10]:  # Subsample for performance
        triangle = np.vstack([tri, tri[0]])
        ax2.plot(triangle[:, 0], triangle[:, 1], 'b-', linewidth=0.3, alpha=0.3)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('Top View (XY)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Front view (XZ plane)
    ax3 = fig.add_subplot(2, 2, 3)
    for tri in vertices[::10]:  # Subsample for performance
        triangle = np.vstack([tri, tri[0]])
        ax3.plot(triangle[:, 0], triangle[:, 2], 'r-', linewidth=0.3, alpha=0.3)
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('Front View (XZ)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # Side view (YZ plane)
    ax4 = fig.add_subplot(2, 2, 4)
    for tri in vertices[::10]:  # Subsample for performance
        triangle = np.vstack([tri, tri[0]])
        ax4.plot(triangle[:, 1], triangle[:, 2], 'g-', linewidth=0.3, alpha=0.3)
    ax4.set_xlabel('Y (mm)')
    ax4.set_ylabel('Z (mm)')
    ax4.set_title('Side View (YZ)')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        print(f"Saving visualization to: {output_file}")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Saved!")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize an STL file")
    parser.add_argument("input", help="Input STL file")
    parser.add_argument("-o", "--output", help="Output image file (PNG)", default=None)

    args = parser.parse_args()

    visualize_stl(args.input, args.output)


if __name__ == "__main__":
    main()
