# Camera Calibration Summary

**Last Updated**: 2025-10-23
**Status**: Final Configuration - Perpendicular Mounting Surface Orientation + 90° CW Rotation

---

## Overview

This document describes the complete camera calibration process for the SO-101 wrist camera, including position calculation from STL geometry, orientation perpendicular to the mounting surface, and field of view optimization.

---

## Camera Mount Geometry Analysis

### STL File
- **File**: `src/lerobot/envs/so101_assets/official_model/assets/Wrist_Cam_Mount_32x32_UVC_Module_SO101_MIRRORED.stl`
- **Coordinate transformation**: 180° rotation around Y-axis applied (quat="0 1 0 0")
- **Mounting surface**: Near-rectangular with 4 corner screw holes + 1 central dodecagon lens hole

### 4 Corner Screw Holes (STL frame)

| Hole | Label | X (mm) | Y (mm) | Z (mm) | Description |
|------|-------|--------|--------|--------|-------------|
| H1 | bottom-left | -12.4 | -61.1 | -10.7 | Corner screw hole |
| H4 | top-left | -13.1 | -85.8 | 1.9 | Corner screw hole |
| H6 | bottom-right | 17.2 | -61.6 | -11.0 | Corner screw hole |
| H9 | top-right | 18.1 | -85.9 | 1.9 | Corner screw hole |

**Mounting surface dimensions**: 30.4mm × 27.6mm (width × height)
**Coplanarity**: All holes within 0.12mm of best-fit plane ✓

### Dodecagon Lens Hole (STL frame)

- **Center**: [2.5, -61.1, -11.1] mm
- **Diameter**: ~28mm
- **Purpose**: Camera lens opening

---

## Camera Position

### Calculation Method
Camera positioned at the center of the dodecagon lens hole (the actual camera lens opening), not at the geometric center of the 4 screw holes.

### Position in Gripper Frame
**Final position**: `[0.0025, 0.0609, 0.0120]` m

**Breakdown**:
- **X**: 2.5 mm (centered in lens hole)
- **Y**: 60.9 mm (distance from gripper base)
- **Z**: 12.0 mm (height above gripper reference)

---

## Camera Orientation

### Calculation Method
1. **Mounting surface plane** calculated from 4 screw hole centers using SVD
2. **Plane normal** extracted (perpendicular to mounting surface)
3. **Direction**: Normal points outward from fixed jaw (toward workspace)
4. **Roll rotation**: Additional 90° clockwise rotation applied for correct image orientation

### Mounting Surface Normal

**STL frame**: `[0.0087, 0.4616, 0.8870]`
**Gripper frame**: `[0.0087, -0.4616, -0.8870]`

**Properties**:
- Perpendicular to mounting plane (27.50° from -Y axis)
- Points outward from fixed jaw
- Verified against all 4 screw holes (coplanar within 0.12mm)

### Final Orientation (with 90° CW roll)

**Axis-angle representation**:
- **Axis**: `[-0.235551, 0.226805, 0.945027]`
- **Angle**: `1.627308` rad = 93.24°

**Rotation breakdown**:
1. Base rotation: 27.50° to align perpendicular to mounting surface
2. Additional roll: 90° clockwise for image orientation
3. Combined: 93.24° total rotation

---

## Field of View (FOV)

### Wrist Camera
- **Current FOV**: 75° (vertical)
- **Rationale**:
  - Matches typical UVC camera modules (70-85° range)
  - Better sim-to-real transfer than wider angles
  - Reduced distortion compared to 90° or 140°
  - Standard for wrist-mounted manipulation cameras
  - More pixels focused on gripper workspace

### Top View Camera
- **Current FOV**: 90° (vertical)
- **Rotation**: 90° clockwise from original
- **Rationale**:
  - Covers full workspace with minimal distortion
  - Standard for overhead manipulation cameras
  - Better than 140° (excessive distortion)

---

## Final XML Configuration

```xml
<!-- Wrist Camera -->
<camera name="wrist_camera"
        pos="0.0025 0.0609 0.0120"
        axisangle="-0.235551 0.226805 0.945027 1.627308"
        fovy="75"/>

<!-- Top View Camera -->
<camera name="top_view"
        pos="0.275 0.175 0.4"
        xyaxes="0 -1 0  1 0 0"
        fovy="90"/>
```

**File location**: `src/lerobot/envs/so101_assets/paper_square_realistic.xml`
- Wrist camera: lines 199-202
- Top view camera: lines 320-323

---

## Verification

### MuJoCo Visualization
When visualized with markers (`scripts/visualize_cameras_with_markers.py`):
- **Position (world frame)**: [0.206, 0.213, 0.260] m
- **Forward direction**: [0.954, 0.301, -0.014] (points mostly in +X, outward from jaw)
- **Markers**: Green sphere + arrow showing camera location and viewing direction

### Geometric Verification
- ✓ Camera at dodecagon lens hole center
- ✓ Orientation perpendicular to mounting surface
- ✓ All 4 screw holes coplanar (0.12mm tolerance)
- ✓ Forward direction points outward from fixed jaw
- ✓ 90° CW roll applied for correct image orientation

---

## Analysis Scripts

### Position & Orientation Calculation
1. **`scripts/find_screw_holes.py`** - Identifies 4 corner screw holes from STL
2. **`scripts/extract_dodecagon_hole.py`** - Extracts central lens hole geometry
3. **`scripts/check_screw_holes_coplanar.py`** - Verifies mounting surface planarity
4. **`scripts/check_rectangle_geometry.py`** - Analyzes mounting surface dimensions
5. **`scripts/calculate_mounting_surface_normal.py`** - Computes perpendicular orientation
6. **`scripts/calculate_rotated_wrist_camera_cw.py`** - Adds 90° CW roll rotation

### Cylinder Axis Analysis
7. **`scripts/analyze_screw_hole_axes.py`** - Calculates screw hole cylinder axes
8. **`scripts/calculate_cylinder_orientations.py`** - Full spatial orientation analysis

### Visualization
9. **`scripts/visualize_cameras_with_markers.py`** - MuJoCo viewer with camera markers

---

## Visualization Images

All images stored in `images/` directory:

1. **`complete_camera_mount_visualization.png`** (1.2 MB)
   - Full camera mount showing BASE, EXTENSION, TIP regions

2. **`four_screw_holes_visualization.png`** (923 KB)
   - 4 corner screw holes (H1, H4, H6, H9) highlighted

3. **`dodecagon_hole_extraction.png`** (1.5 MB)
   - Central dodecagon lens hole with 28mm diameter

4. **`screw_holes_analysis.png`** (655 KB)
   - Geometric analysis of screw hole positions

5. **`mounting_surface_visualization.png`** (597 KB)
   - Mounting surface plane with all 5 holes

---

## Video Demonstrations

All videos stored in `videos/` directory:

### Dual-Feed Videos (Top View + Wrist Camera)

1. **`groot_n1.5_dual_feed_updated_camera.mp4`** (6.0 MB)
   - 140° FOV, original orientation (perpendicular to mounting surface)

2. **`groot_n1.5_dual_feed_fov90.mp4`** (7.2 MB)
   - 90° FOV for both cameras, no rotation

3. **`groot_n1.5_dual_feed_fov90_cw.mp4`** (7.4 MB)
   - 90° FOV + 90° CW rotation (wrist: 90° FOV)

4. **Latest**: 75° FOV wrist camera (generate new video to demonstrate)

All videos show GR00T N1.5 policy controlling SO-101 for paper-return task.

---

## Configuration History

### Version 1: Initial (Before Calibration)
- Position: Approximate/estimated
- Orientation: Not calibrated
- FOV: 140° (both cameras)

### Version 2: Screw Hole Center (2025-10-23)
- Position: Geometric center of 4 screw holes
- Orientation: Along cylinder bore axes (~25° from Y)
- FOV: 140°

### Version 3: Dodecagon Center (2025-10-23)
- Position: Center of dodecagon lens hole ✓
- Orientation: Still along cylinder axes
- FOV: 140°

### Version 4: Perpendicular to Surface (2025-10-23)
- Position: Dodecagon lens hole center ✓
- Orientation: Perpendicular to mounting surface ✓
- FOV: 140°

### Version 5: Optimized FOV (2025-10-23)
- Position: Dodecagon lens hole center ✓
- Orientation: Perpendicular to mounting surface ✓
- FOV: 90° (both cameras)

### Version 6: Final - Current (2025-10-23) ✓
- **Position**: Dodecagon lens hole center ✓
- **Orientation**: Perpendicular + 90° CW roll ✓
- **FOV**:
  - Wrist camera: **75°** (realistic, matches hardware)
  - Top view: **90°** (optimized workspace coverage)
- **Rotation**: Both cameras rotated 90° clockwise

---

## Technical Notes

### Coordinate Frame Transformations
1. **STL → Gripper**: 180° Y-rotation (quat="0 1 0 0")
   - Transformation: `(x, y, z)_STL → (x, -y, -z)_gripper`

2. **Gripper → World**: Standard MuJoCo body transformations
   - Depends on gripper pose in simulation

### Mounting Surface Geometry
- **Type**: Near-rectangle (not perfect)
- **Dimensions**: 30.4mm × 27.6mm
- **Edge differences**: Top/bottom differ by 1.6mm
- **Corner angles**: 87.96° - 92.46° (average 90°)
- **Coplanarity**: Excellent (0.12mm max deviation)

### Camera Roll Rotation
- **Purpose**: Correct image orientation in rendered views
- **Direction**: 90° clockwise around viewing axis
- **Effect**: Camera's "up" vector rotates to align with desired image orientation
- **Implementation**: Composed with base perpendicular orientation

---

## References

### LeRobot Documentation
- Camera configuration: `docs/source/cameras.mdx`
- SO-101 robot: `src/lerobot/robots/so101_mujoco/README.md`

### External Resources
- MuJoCo camera documentation: https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera
- Typical camera FOV ranges: 70-90° for wrist cameras, 90-110° for overhead

---

## Summary

The wrist camera is now:
1. ✅ **Positioned** at the center of the dodecagon lens hole (actual camera opening)
2. ✅ **Oriented** perpendicular to the mounting surface (outward from fixed jaw)
3. ✅ **Rotated** 90° clockwise for correct image orientation
4. ✅ **Configured** with 75° FOV (realistic, matches typical UVC modules)
5. ✅ **Verified** through geometric analysis and MuJoCo visualization

This configuration provides accurate camera positioning for the SO-101 robot simulation and should provide good sim-to-real transfer characteristics.
