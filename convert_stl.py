#!/usr/bin/env python
"""Convert ASCII STL files to binary format for MuJoCo compatibility."""

from stl import mesh
import sys
from pathlib import Path

def convert_stl_to_binary(input_path, output_path=None):
    """Convert ASCII STL to binary format."""
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix('.binary.stl')

    # Load the ASCII STL
    stl_mesh = mesh.Mesh.from_file(str(input_path))

    # Save as binary
    stl_mesh.save(str(output_path), mode=mesh.Mode.BINARY)

    print(f"Converted: {input_path} -> {output_path}")
    return output_path

if __name__ == "__main__":
    # Convert the problematic STL file
    stl_path = "src/lerobot/envs/so101_assets/official_model/assets/waveshare_mounting_plate_so101_v2.stl"

    # Create backup
    import shutil
    backup_path = stl_path + ".ascii_backup"
    if not Path(backup_path).exists():
        shutil.copy(stl_path, backup_path)
        print(f"Created backup: {backup_path}")

    # Convert and overwrite original
    temp_output = stl_path + ".binary"
    convert_stl_to_binary(stl_path, temp_output)

    # Replace original with binary version
    shutil.move(temp_output, stl_path)
    print(f"Replaced {stl_path} with binary version")