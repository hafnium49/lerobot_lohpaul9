#!/usr/bin/env python3
"""
Fix camera orientation to point toward workspace (opposite of surface normal).
"""

import numpy as np


print("="*80)
print("Corrected Camera Orientation Calculation")
print("="*80)

# Surface normal in gripper frame (perpendicular to mounting surface, pointing outward)
surface_normal_gripper = np.array([0.2276068, -0.88252636, 0.41151229])

print(f"\nSurface normal (outward from mount): {surface_normal_gripper}")

# Camera should point OPPOSITE to surface normal (inward, toward workspace)
camera_forward = -surface_normal_gripper

print(f"Camera forward direction (toward workspace): {camera_forward}")

# Convert to euler angles
# Camera's -Z axis should point along camera_forward
# Yaw (around Z): rotation in XY plane
yaw = np.arctan2(camera_forward[0], camera_forward[1])
# Pitch (around X): rotation from horizontal
pitch = np.arctan2(-camera_forward[2],
                   np.sqrt(camera_forward[0]**2 + camera_forward[1]**2))

print(f"\nEuler angles (yaw, pitch, roll):")
print(f"  Radians: [{yaw:.4f}, {pitch:.4f}, 0]")
print(f"  Degrees: [{np.degrees(yaw):.1f}°, {np.degrees(pitch):.1f}°, 0°]")
print(f"  Formatted: euler=\"{yaw:.4f} {pitch:.4f} 0\"")

print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)
print(f"\n<camera name=\"wrist_camera\"")
print(f"        pos=\"0.0024 0.0783 0.0013\"")
print(f"        euler=\"{yaw:.4f} {pitch:.4f} 0\"")
print(f"        fovy=\"140\"/>")
print("\n" + "="*80)
