"""
test_disk.py
------------
Unit tests for solar disk geometry detection and masking.
"""

import numpy as np
import pytest
from sunspot.disk import create_disk_mask, scale_geometry, detect_disk_classical


def test_create_disk_mask():
    shape = (100, 100)
    center = (50.0, 50.0)
    radius = 40.0
    mask = create_disk_mask(shape, center, radius, r_fraction=1.0)

    assert mask.shape == shape
    assert mask.dtype == bool
    # Center pixel must be inside
    assert mask[50, 50] is True or mask[50, 50] == 1
    # Corner pixel must be outside
    assert mask[0, 0] is False or mask[0, 0] == 0
    # Area should be approximately pi * r^2
    expected_area = np.pi * (radius ** 2)
    assert np.isclose(mask.sum(), expected_area, rtol=0.05)


def test_scale_geometry():
    geom = (2000.0, 2000.0, 1900.0)
    scaled = scale_geometry(geom, 0.25)
    assert scaled == (500.0, 500.0, 475.0)


def test_detect_disk_classical():
    # Create synthetic circular disk
    img = np.zeros((200, 200), dtype=np.float32)
    y, x = np.ogrid[:200, :200]
    circle = ((x - 100) ** 2 + (y - 100) ** 2) <= (60 ** 2)
    img[circle] = 1000.0

    cx, cy, r = detect_disk_classical(img)
    assert np.isclose(cx, 100.0, atol=2.0)
    assert np.isclose(cy, 100.0, atol=2.0)
    assert np.isclose(r, 60.0, atol=2.0)
