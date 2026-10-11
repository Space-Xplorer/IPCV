"""
test_features.py
----------------
Unit tests for connected components feature extraction.
"""

import numpy as np
import pytest
from sunspot.features import extract_features


def test_extract_features_synthetic():
    mask = np.zeros((100, 100), dtype=np.uint8)
    norm = np.ones((100, 100), dtype=np.float32)

    # Component 1: 10x10 square (100 pixels)
    mask[10:20, 10:20] = 1
    norm[10:20, 10:20] = 0.5

    # Component 2: 5x5 square (25 pixels)
    mask[50:55, 50:55] = 1
    norm[50:55, 50:55] = 0.3

    disk_radius = 45.0
    df, summary = extract_features(mask, norm, disk_radius)

    assert summary["num_spots"] == 2
    assert summary["total_area_pixels"] == 125
    assert len(df) == 2

    # Check bounds of geometric features
    for idx, row in df.iterrows():
        assert 0.0 <= row["eccentricity"] <= 1.0
        assert 0.0 <= row["solidity"] <= 1.0
        assert row["mean_intensity"] < 1.0

    # Test with umbra mask
    umbra_mask = np.zeros_like(mask)
    umbra_mask[12:18, 12:18] = 1  # 36 pixels of umbra in component 1
    df_u, summary_u = extract_features(mask, norm, disk_radius, umbra_mask=umbra_mask)
    assert summary_u["total_umbra_msh"] > 0
    assert summary_u["total_penumbra_msh"] > 0
    assert df_u.loc[0, "umbra_pixels"] == 36
    assert df_u.loc[0, "penumbra_pixels"] == 64


def test_extract_features_empty_mask():
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    norm = np.ones((100, 100), dtype=np.float32)
    disk_radius = 45.0
    df, summary = extract_features(empty_mask, norm, disk_radius)
    assert len(df) == 0
    assert summary["num_spots"] == 0
    assert summary["total_area_pixels"] == 0
    assert summary["total_area_msh"] == 0.0


