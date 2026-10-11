"""
cleanup.py
----------
Morphological cleanup and connected-component filtering.

Implements:
1. Small structuring element binary opening to sever single-pixel noise bridges.
2. Binary hole filling to consolidate fractured internal structures.
3. Connected-component minimum-area pore rejection (adapted from Reference 2).
"""

import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes


def cleanup_mask(
    raw_mask: np.ndarray,
    min_area_pixels: int = 5,
    apply_opening: bool = True,
    fill_holes: bool = True
) -> np.ndarray:
    """
    Apply morphological cleanup and minimum-area pore filtering.

    Parameters
    ----------
    raw_mask : np.ndarray
        Binary uint8 mask from initial segmentation.
    min_area_pixels : int, default 5
        Minimum component area in pixels (at 1024x1024, 5 px corresponds to ~12 MSH).
        Connected regions below this area are classified as pores and rejected.
    apply_opening : bool, default True
        Whether to apply 3x3 binary opening to sever thin single-pixel noise bridges.
    fill_holes : bool, default True
        Whether to fill small internal holes inside detected sunspots.

    Returns
    -------
    clean_mask : np.ndarray
        Refined binary uint8 mask.
    """
    cleaned = raw_mask.copy().astype(np.uint8)

    # Step 1: Binary Opening (sever thin single-pixel noise bridges)
    if apply_opening:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # Step 2: Binary Hole Filling
    if fill_holes:
        cleaned = binary_fill_holes(cleaned).astype(np.uint8)

    # Step 3: Connected-Component Minimum Area Pore Filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    clean_mask = np.zeros_like(cleaned, dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area_pixels:
            clean_mask[labels == i] = 1

    return clean_mask
