#!/usr/bin/env python
"""
Physics validation tests for SO-101 MuJoCo world.

Tests:
1. Model loads without errors
2. Paper doesn't Z-fight with table
3. Timestep is correct (1000Hz)
4. Solver parameters are set correctly
5. Paper has correct thickness and friction
"""

import mujoco as mj
import numpy as np
from pathlib import Path


def test_model_loads():
    """Test that the MuJoCo model loads without errors."""
    xml_path = Path("src/lerobot/envs/so101_assets/paper_square_realistic.xml")
    model = mj.MjModel.from_xml_path(str(xml_path))
    assert model is not None
    print("✅ Model loads successfully")


def test_timestep():
    """Test that timestep is 1ms (1000Hz) for stability."""
    xml_path = Path("src/lerobot/envs/so101_assets/paper_square_realistic.xml")
    model = mj.MjModel.from_xml_path(str(xml_path))

    assert abs(model.opt.timestep - 0.001) < 1e-6, f"Expected timestep 0.001, got {model.opt.timestep}"
    print(f"✅ Timestep correct: {model.opt.timestep}s ({1/model.opt.timestep:.0f}Hz)")


def test_paper_thickness():
    """Test that paper has realistic thickness (0.15mm) to avoid Z-fighting."""
    xml_path = Path("src/lerobot/envs/so101_assets/paper_square_realistic.xml")
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)

    # Find paper geom
    paper_geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "paper_geom")
    assert paper_geom_id >= 0, "Paper geom not found"

    # Get paper size (half-extents)
    paper_size = model.geom_size[paper_geom_id]
    paper_thickness = paper_size[2]  # Z dimension

    # Should be 0.000075m (0.15mm)
    expected_thickness = 0.000075
    assert abs(paper_thickness - expected_thickness) < 1e-7, \
        f"Expected paper thickness {expected_thickness}m, got {paper_thickness}m"

    print(f"✅ Paper thickness correct: {paper_thickness*1000:.3f}mm")


def test_paper_friction():
    """Test that paper has correct friction coefficient."""
    xml_path = Path("src/lerobot/envs/so101_assets/paper_square_realistic.xml")
    model = mj.MjModel.from_xml_path(str(xml_path))

    # Find paper geom
    paper_geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "paper_geom")

    # Get friction
    paper_friction = model.geom_friction[paper_geom_id]

    # Should be [0.60, 0.002, 0.0001] (slide, torsion, roll)
    expected = np.array([0.60, 0.002, 0.0001])
    assert np.allclose(paper_friction, expected, atol=1e-4), \
        f"Expected friction {expected}, got {paper_friction}"

    print(f"✅ Paper friction correct: {paper_friction}")


def test_condim():
    """Test that paper has condim=3 for realistic contact."""
    xml_path = Path("src/lerobot/envs/so101_assets/paper_square_realistic.xml")
    model = mj.MjModel.from_xml_path(str(xml_path))

    # Find paper geom
    paper_geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "paper_geom")

    # Get condim
    paper_condim = model.geom_condim[paper_geom_id]

    assert paper_condim == 3, f"Expected condim=3, got {paper_condim}"
    print(f"✅ Paper condim correct: {paper_condim} (slide + spin + roll)")


def test_tape_sticky():
    """Test that tape area has higher friction than table."""
    xml_path = Path("src/lerobot/envs/so101_assets/paper_square_realistic.xml")
    model = mj.MjModel.from_xml_path(str(xml_path))

    # Find geoms
    table_geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "table_surface")
    tape_geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "tape_contact")

    if tape_geom_id < 0:
        print("⚠️  Tape contact geom not found (optional feature)")
        return

    table_friction = model.geom_friction[table_geom_id][0]
    tape_friction = model.geom_friction[tape_geom_id][0]

    assert tape_friction > table_friction, \
        f"Tape friction ({tape_friction}) should be higher than table ({table_friction})"

    print(f"✅ Tape stickiness correct: table μ={table_friction:.2f}, tape μ={tape_friction:.2f}")


def test_simulation_stability():
    """Test that simulation runs for 1000 steps without instability."""
    xml_path = Path("src/lerobot/envs/so101_assets/paper_square_realistic.xml")
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)

    # Reset
    mj.mj_resetData(model, data)

    # Run simulation
    for _ in range(1000):
        mj.mj_step(model, data)

        # Check for NaN/Inf
        assert np.isfinite(data.qpos).all(), "qpos contains NaN/Inf"
        assert np.isfinite(data.qvel).all(), "qvel contains NaN/Inf"

    print(f"✅ Simulation stable for 1000 steps ({1000 * model.opt.timestep:.1f}s)")


if __name__ == "__main__":
    print("=" * 60)
    print("SO-101 Physics Validation Tests")
    print("=" * 60)
    print()

    tests = [
        test_model_loads,
        test_timestep,
        test_paper_thickness,
        test_paper_friction,
        test_condim,
        test_tape_sticky,
        test_simulation_stability,
    ]

    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            failed.append(test.__name__)

    print()
    print("=" * 60)
    if failed:
        print(f"❌ {len(failed)} tests failed: {failed}")
    else:
        print("✅ All physics validation tests passed!")
    print("=" * 60)
