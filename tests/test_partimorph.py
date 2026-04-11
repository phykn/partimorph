import numpy as np
import pytest
import partimorph as pm


def test_create_particle_mask():
    """Test generating a synthetic particle mask."""
    shape = (256, 256)
    radius = 50
    mask = pm.utils.create_particle_mask(
        shape=shape,
        center=(128, 128),
        radius=radius,
        sphericity=0.8,
        roundness=0.7,
    )
    assert isinstance(mask, np.ndarray)
    assert mask.shape == shape
    assert mask.dtype == bool
    assert np.any(mask)
    # Check if the particle is roughly in the center
    y_coords, x_coords = np.where(mask)
    assert np.abs(np.mean(y_coords) - 128) < 5
    assert np.abs(np.mean(x_coords) - 128) < 5


def test_analyze_mask_full():
    """Test the main end-to-end analysis function."""
    # Create a simple square mask
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True

    results = pm.analyze_mask(mask)

    assert results is not None
    assert "roundness" in results
    assert "circularity" in results
    assert "sphericity" in results
    assert "aspect_ratio" in results

    # Square should have aspect ratio approx 1.0
    assert results["aspect_ratio"] is not None
    assert np.isclose(results["aspect_ratio"]["val"], 1.0, atol=0.1)


def test_to_binary_validation():
    """Test the input validation and normalization utility."""
    from partimorph.validation import to_binary

    # Valid bool mask
    bool_mask = np.zeros((10, 10), dtype=bool)
    result = to_binary(bool_mask)
    assert result.dtype == np.uint8
    assert np.all(np.isin(result, [0, 1]))

    # Valid int mask with 0 and 1
    int_mask = np.array([[0, 1], [1, 0]], dtype=np.int32)
    result = to_binary(int_mask)
    assert result.dtype == np.uint8

    # Invalid values should raise ValueError
    with pytest.raises(ValueError, match="binary with values in {0, 1}"):
        to_binary(np.array([[0, 2], [1, 0]]))

    # 3D array should raise ValueError
    with pytest.raises(ValueError, match="mask must be 2D"):
        to_binary(np.zeros((10, 10, 10)))


def test_edge_case_zero_radius():
    """Test that zero radius yields a single pixel mask or empty info."""
    shape = (100, 100)
    mask, info = pm.utils.create_particle_mask(
        shape=shape,
        center=(50, 50),
        radius=0.0,
        sphericity=0.8,
        roundness=0.8,
        return_info=True,
    )
    assert np.count_nonzero(mask) <= 1
    assert info["roundness_achieved"] == 0.0


def test_metrics_consistency():
    """Check that results are consistent metrics-wise."""
    # Circle (approximated)
    y, x = np.ogrid[:100, :100]
    mask = (x - 50) ** 2 + (y - 50) ** 2 <= 40**2

    results = pm.analyze_mask(mask)

    # For a circle, all morphological indices should be near 1.0
    assert results["circularity"]["val"] > 0.9
    assert results["sphericity"]["val"] > 0.9
    assert results["roundness"]["val"] > 0.9


def test_analyze_mask_invalid_params():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True

    with pytest.raises(ValueError, match="eps must be > 0"):
        pm.analyze_mask(mask, eps=0.0)

    with pytest.raises(ValueError, match="target_dim must be > 0"):
        pm.analyze_mask(mask, target_dim=0)


def test_analyze_mask_uses_largest_component():
    mask = np.zeros((120, 120), dtype=bool)
    mask[10:30, 10:30] = True
    mask[50:110, 50:110] = True

    results = pm.analyze_mask(mask, use_roundness=False, use_circularity=False)
    assert results is not None
    assert results["sphericity"] is not None

    center = results["sphericity"]["inscribed"]
    assert isinstance(center["x"], float)
    assert isinstance(center["y"], float)
    assert abs(center["x"] - 80) < 5
    assert abs(center["y"] - 80) < 5


def test_analyze_mask_resize_path_scales_coordinates():
    mask = np.zeros((1000, 1000), dtype=bool)
    mask[100:900, 200:800] = True

    results = pm.analyze_mask(mask, target_dim=384, use_roundness=False)
    assert results is not None
    assert results["sphericity"] is not None
    assert results["aspect_ratio"] is not None

    inscribed = results["sphericity"]["inscribed"]
    assert isinstance(inscribed["x"], float)
    assert isinstance(inscribed["y"], float)
    assert abs(inscribed["x"] - 500) < 5
    assert abs(inscribed["y"] - 500) < 110

    ellipse = results["aspect_ratio"]["ellipse"]
    assert len(ellipse["bbox"]) == 4
    for pt in ellipse["bbox"]:
        assert len(pt) == 2
        assert isinstance(pt[0], float)
        assert isinstance(pt[1], float)


def test_roundness_rough_particle_golden():
    mask = pm.utils.create_particle_mask(
        shape=(256, 256),
        center=(128, 128),
        radius=60,
        sphericity=0.85,
        roundness=0.5,
        seed=42,
    )
    results = pm.analyze_mask(mask)
    assert results is not None
    assert results["roundness"] is not None
    value = results["roundness"]["val"]
    assert 0.3 < value < 0.7
