"""
test_normalize.py
-----------------
Unit tests for solar image normalization, downsampling, and limb-darkening correction.
"""

import numpy as np
import pytest
from sunspot.normalize import downsample_image, compute_radial_profile, correct_limb_darkening
from sunspot.disk import create_disk_mask


def test_downsample_image():
    img = np.ones((400, 400), dtype=np.float32)
    small, scale = downsample_image(img, target_size=100)
    assert small.shape == (100, 100)
    assert scale == 0.25


def test_correct_limb_darkening_synthetic():
    # Synthetic disk with quadratic limb darkening: I(rho) = 1000 * (1 - 0.4 * rho^2)
    shape = (200, 200)
    center = (100.0, 100.0)
    radius = 80.0
    mask = create_disk_mask(shape, center, radius, r_fraction=0.98)

    y, x = np.ogrid[:200, :200]
    dist = np.sqrt((x - 100) ** 2 + (y - 100) ** 2)
    rho = dist / radius

    synth_img = np.zeros(shape, dtype=np.float32)
    synth_img[mask] = 1000.0 * (1.0 - 0.4 * (rho[mask] ** 2))

    norm_img, poly = correct_limb_darkening(synth_img, mask, center, radius, apply_median_filter=False)

    # After normalization, disk pixels should be uniformly close to 1.0
    disk_vals = norm_img[mask]
    assert np.isclose(np.mean(disk_vals), 1.0, atol=0.03)
    assert np.isclose(np.std(disk_vals), 0.0, atol=0.03)
    # Off-disk pixels must be exactly zero
    assert np.all(norm_img[~mask] == 0.0)
