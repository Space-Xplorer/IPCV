"""
test_segment.py
---------------
Unit tests for segmentation algorithms and morphological cleanup.
"""

import numpy as np
import pytest
from sunspot.segment import segment_tophat, segment_otsu, segment_adaptive
from sunspot.cleanup import cleanup_mask
from sunspot.disk import create_disk_mask


def test_segment_tophat_synthetic_spot():
    # Synthetic flat quiet-Sun disk (~1.0) with an artificial dark spot (intensity ~0.4)
    shape = (100, 100)
    center = (50.0, 50.0)
    radius = 45.0
    disk_mask = create_disk_mask(shape, center, radius, r_fraction=0.98)

    norm = np.zeros(shape, dtype=np.float32)
    norm[disk_mask] = 1.0  # Quiet Sun

    # Insert a dark circular spot at (50, 50) of radius 5
    y, x = np.ogrid[:100, :100]
    spot = ((x - 50) ** 2 + (y - 50) ** 2) <= (5 ** 2)
    norm[spot] = 0.4  # Sunspot

    mask, res, umbra = segment_tophat(norm, disk_mask, radius=10, k_sigma=2.0)
    clean = cleanup_mask(mask, min_area_pixels=5)

    # Spot center must be detected
    assert clean[50, 50] == 1
    # Spot border must be detected
    assert clean.sum() > 20
    # Far-away quiet Sun pixel must not be detected
    assert clean[20, 20] == 0


def test_cleanup_mask_removes_noise():
    raw = np.zeros((50, 50), dtype=np.uint8)
    # 2 isolated 1-pixel noise artifacts
    raw[5, 5] = 1
    raw[45, 45] = 1
    # One legitimate component of 20 pixels
    raw[20:25, 20:24] = 1

    clean = cleanup_mask(raw, min_area_pixels=8, apply_opening=True)
    assert clean[5, 5] == 0
    assert clean[45, 45] == 0
    assert clean[20:25, 20:24].sum() >= 15


def test_decompose_umbra_penumbra():
    from sunspot.segment import decompose_umbra_penumbra
    # Synthetic image: quiet Sun ~ 1.0, penumbra ~ 0.80, umbra ~ 0.50
    norm = np.ones((40, 40), dtype=np.float32)
    spot_mask = np.zeros((40, 40), dtype=np.uint8)

    # Penumbra region: 10x10 square
    spot_mask[15:25, 15:25] = 1
    norm[15:25, 15:25] = 0.80

    # Umbra core: 4x4 square inside penumbra
    norm[18:22, 18:22] = 0.50

    umbra_mask, penumbra_mask = decompose_umbra_penumbra(norm, spot_mask, umbra_threshold=0.72)

    # Core must be umbra
    assert umbra_mask[20, 20] == 1
    # Outer spot pixel must be penumbra
    assert penumbra_mask[16, 16] == 1
    assert umbra_mask[16, 16] == 0
    # Together they must partition the spot mask
    assert np.array_equal((umbra_mask | penumbra_mask), spot_mask)

