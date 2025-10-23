# Camera Calibration Summary

## Analysis Method

We analyzed the camera mount STL file to find the 4 corner screw holes and calculated the camera position at their geometric center.

## 4 Corner Screw Holes Found (STL frame)

| Hole | Label | X (m) | Y (m) | Z (m) | X (mm) | Y (mm) | Z (mm) |
|------|-------|-------|-------|-------|--------|--------|--------|
| H1 | bottom-left | -0.0124 | -0.0611 | -0.0107 | -12.4 | -61.1 | -10.7 |
| H4 | top-left | -0.0131 | -0.0858 | 0.0019 | -13.1 | -85.8 | 1.9 |
| H6 | bottom-right | 0.0172 | -0.0616 | -0.0110 | 17.2 | -61.6 | -11.0 |
| H9 | top-right | 0.0181 | -0.0859 | 0.0019 | 18.1 | -85.9 | 1.9 |

## Screw Hole Spacing

- **X (left-right)**: 31.2 mm
- **Y (depth)**: 24.8 mm
- **Z (up-down)**: 12.9 mm

## Camera Position Calculation

### STL Frame
- **Geometric center**: [0.0024, -0.0736, -0.0045] m
- **At surface** (Y_max): [0.0024, -0.0611, -0.0045] m

### Transformation to Gripper Frame
- **Camera mount position**: [0, -0.000218214, 0.000949706] m
- **Camera mount rotation**: quat="0 1 0 0" (180° around Y-axis)
- **Transformation**: (x, y, z) → (x, -y, -z) + mount_pos

### Result in Gripper Frame
**Camera position**: [0.0024, 0.0609, 0.0054] m

## Camera Orientation

### Surface Normal (STL frame)
- Direction: [0, 0.9063, -0.4226]
- Tilt: 25° from Y-axis

### Camera Forward (Gripper frame)
- Direction: [0, 0.9063, -0.4226] (points into workspace)
- Perpendicular to mounting surface

### Axis-Angle Representation
- **Axis**: [1.0, 0.0, 0.0] (X-axis)
- **Angle**: 1.134477 rad = 65.0°

## Final XML Configuration

```xml
<camera name="wrist_camera"
        pos="0.0024 0.0609 0.0054"
        axisangle="1.000000 0.000000 0.000000 1.134477"
        fovy="140"/>
```

## Verification

### In World Frame (MuJoCo simulation)
When visualized in MuJoCo with markers:
- **Position**: [0.211, 0.217, 0.260] m
- **Forward direction**: [-0.341, 0.939, 0.044]

### Location
- At the geometric center of 4 corner screw holes
- On the front surface of the mounting plate
- Inside the dodecagon lens hole opening

## Files

- **STL file**: `src/lerobot/envs/so101_assets/official_model/assets/Wrist_Cam_Mount_32x32_UVC_Module_SO101_MIRRORED.stl`
- **XML config**: `src/lerobot/envs/so101_assets/paper_square_realistic.xml` (lines 193-196)
- **Analysis script**: `scripts/find_screw_holes.py`
- **Calculation script**: `scripts/calculate_camera_from_screw_holes.py`
- **Visualization**: `complete_camera_mount_visualization.png`

## Notes

- The mounting surface is rectangular (~31mm × 13mm), not square (32mm × 32mm)
- The Z-dimension is smaller because the STL only models the rim around the holes
- Camera faces toward the fingertip/workspace, tilted 25° from perpendicular
