"""
normalize.py
------------
Solar image preprocessing, spatial downsampling, and limb-darkening correction.

Implements:
1. Spatial downsampling (4096x4096 -> 1024x1024) with area anti-aliasing.
2. Median spatial filtering for cosmic ray and high-frequency noise suppression.
3. Radial quiet-Sun profile calculation using robust annular bin medians.
4. 5th-order polynomial limb-darkening fitting and normalization.
"""

from typing import Tuple
import numpy as np
import cv2
from scipy.ndimage import median_filter


def downsample_image(
    image: np.ndarray,
    target_size: int = 1024
) -> Tuple[np.ndarray, float]:
    """
    Downscale a 2D solar float array to target_size (square).

    Parameters
    ----------
    image : np.ndarray
        Raw 2D float array (e.g. 4096 x 4096).
    target_size : int, default 1024
        Target resolution in pixels.

    Returns
    -------
    downsampled : np.ndarray
        Resized 2D float array.
    scale_factor : float
        Ratio of target_size to original height.
    """
    h, w = image.shape
    scale_factor = float(target_size) / float(max(h, w))
    clean_float = np.nan_to_num(image, nan=0.0).astype(np.float32)
    resized = cv2.resize(
        clean_float,
        (target_size, target_size),
        interpolation=cv2.INTER_AREA
    )
    return resized, scale_factor


def compute_radial_profile(
    image: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    num_bins: int = 150,
    r_max_frac: float = 0.92
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute quiet-Sun radial intensity profile using annular median bins.

    Parameters
    ----------
    image : np.ndarray
        2D intensity array.
    center : tuple (cx, cy)
        Disk center coordinates.
    radius : float
        Disk radius in pixels.
    num_bins : int, default 150
        Number of concentric radial bins.
    r_max_frac : float, default 0.98
        Maximum normalized radius to include.

    Returns
    -------
    bin_centers : np.ndarray
        Normalized radial distances rho = r / R_sun (from 0.0 to r_max_frac).
    bin_medians : np.ndarray
        Median intensity at each radial bin.
    """
    h, w = image.shape
    y_coords, x_coords = np.ogrid[:h, :w]
    cx, cy = center
    dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    rho = dist / radius

    # Filter to valid disk pixels within r_max_frac
    valid_mask = (rho <= r_max_frac) & (image > 0)
    rho_valid = rho[valid_mask]
    img_valid = image[valid_mask]

    bin_edges = np.linspace(0.0, r_max_frac, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_indices = np.digitize(rho_valid, bin_edges) - 1

    bin_medians = np.zeros(num_bins, dtype=np.float32)
    for b in range(num_bins):
        vals = img_valid[bin_indices == b]
        if len(vals) > 0:
            bin_medians[b] = float(np.median(vals))
        else:
            bin_medians[b] = bin_medians[b - 1] if b > 0 else 0.0

    return bin_centers, bin_medians


def fit_limb_darkening_polynomial(
    bin_centers: np.ndarray,
    bin_medians: np.ndarray,
    degree: int = 5
) -> np.poly1d:
    """
    Fit polynomial to quiet-Sun radial intensity profile.

    Parameters
    ----------
    bin_centers : np.ndarray
        Normalized radial positions rho.
    bin_medians : np.ndarray
        Quiet-Sun median intensity per bin.
    degree : int, default 5
        Polynomial degree.

    Returns
    -------
    poly_func : np.poly1d
        Fitted polynomial function I(rho).
    """
    valid = bin_medians > 0
    coeffs = np.polyfit(bin_centers[valid], bin_medians[valid], deg=degree)
    return np.poly1d(coeffs)


def correct_limb_darkening(
    image: np.ndarray,
    disk_mask: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    apply_median_filter: bool = True
) -> Tuple[np.ndarray, np.poly1d]:
    """
    Remove solar limb darkening and normalize quiet-Sun intensity to ~1.0.

    Parameters
    ----------
    image : np.ndarray
        Downsampled float intensity image.
    disk_mask : np.ndarray
        Boolean circular mask of the solar disk.
    center : tuple (cx, cy)
        Disk center coordinates.
    radius : float
        Disk radius in pixels.
    apply_median_filter : bool, default True
        Whether to apply 3x3 median filtering before radial fitting.

    Returns
    -------
    normalized : np.ndarray
        Normalized image where quiet-Sun photosphere is ~1.0, sunspots < 1.0,
        and off-limb pixels are 0.0.
    poly_fit : np.poly1d
        The fitted limb-darkening radial polynomial function.
    """
    # Step 1: Pre-filter noise if requested
    filtered = cv2.medianBlur(image, 3) if apply_median_filter else image.copy()

    # Step 2: Compute radial median profile
    bin_centers, bin_medians = compute_radial_profile(filtered, center, radius)

    # Step 3: Fit polynomial
    poly = fit_limb_darkening_polynomial(bin_centers, bin_medians, degree=5)

    # Step 4: Evaluate polynomial across the 2D grid
    h, w = image.shape
    y_coords, x_coords = np.ogrid[:h, :w]
    cx, cy = center
    dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    rho = dist / radius

    expected_intensity = poly(rho)
    # Guard against zero or negative division
    expected_intensity = np.maximum(expected_intensity, 1e-4)

    # Step 5: Normalize disk pixels and zero out off-disk background
    normalized = np.zeros_like(image, dtype=np.float32)
    normalized[disk_mask] = filtered[disk_mask] / expected_intensity[disk_mask]
    normalized[~disk_mask] = 0.0

    return normalized, poly
