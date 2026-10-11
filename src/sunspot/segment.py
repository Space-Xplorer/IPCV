"""
segment.py
----------
Multi-method classical segmentation suite for full-disk solar continuum images.

Implements three classical paradigms for systematic benchmark comparison:
1. Primary: Morphological White Top-Hat transform with statistical cutoff
   (adapted and accelerated from Reference 2).
2. Baseline 1: Global Otsu thresholding within the solar disk mask.
3. Baseline 2: Local adaptive Gaussian thresholding within the solar disk mask.
"""

from typing import Tuple, Optional
import numpy as np
import cv2
from skimage.filters import threshold_otsu


def decompose_umbra_penumbra(
    norm_image: np.ndarray,
    spot_mask: np.ndarray,
    umbra_threshold: float = 0.72,
    adaptive_local: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hierarchical morphological decomposition of sunspots into umbra (dark core)
    and penumbra (diffuse intermediate halo).

    Parameters
    ----------
    norm_image : np.ndarray
        Limb-darkening normalized float image (quiet Sun ~ 1.0).
    spot_mask : np.ndarray
        Cleaned binary uint8 sunspot mask (1 = sunspot, 0 = background).
    umbra_threshold : float, default 0.72
        Intensity cutoff for umbral core (I_norm < 0.72 corresponds to deep magnetic depression).
    adaptive_local : bool, default True
        Whether to refine umbra threshold per connected component using component statistics.

    Returns
    -------
    umbra_mask : np.ndarray
        Binary uint8 mask of umbral regions (1 = umbra, 0 = other).
    penumbra_mask : np.ndarray
        Binary uint8 mask of penumbral regions (1 = penumbra, 0 = other).
    """
    umbra_mask = np.zeros_like(spot_mask, dtype=np.uint8)
    
    if not np.any(spot_mask):
        penumbra_mask = np.zeros_like(spot_mask, dtype=np.uint8)
        return umbra_mask, penumbra_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(spot_mask, connectivity=8)
    
    for i in range(1, num_labels):
        comp_mask = (labels == i)
        comp_pixels = norm_image[comp_mask]
        if len(comp_pixels) == 0:
            continue
            
        if adaptive_local and len(comp_pixels) >= 12:
            # Multi-level adaptive threshold within the individual spot component:
            # tau_u is bounded by umbra_threshold and local mean - 0.5 * std
            mu_c = float(np.mean(comp_pixels))
            sigma_c = float(np.std(comp_pixels))
            local_tau = min(umbra_threshold, mu_c - 0.5 * sigma_c)
            # Ensure local_tau is strictly below quiet-sun penumbral values
            local_tau = max(0.40, min(local_tau, umbra_threshold))
            u_pixels = comp_mask & (norm_image < local_tau)
        else:
            u_pixels = comp_mask & (norm_image < umbra_threshold)
            
        umbra_mask[u_pixels] = 1

    # Morphological hole filling and noise opening on umbra within spot boundaries
    umbra_mask = (cv2.morphologyEx(umbra_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) & spot_mask).astype(np.uint8)
    
    # Penumbra is strictly the spot region minus the umbral core
    penumbra_mask = ((spot_mask > 0) & (umbra_mask == 0)).astype(np.uint8)

    return umbra_mask, penumbra_mask


def segment_tophat(
    norm_image: np.ndarray,
    disk_mask: np.ndarray,
    radius: int = 21,
    k_sigma: float = 2.0,
    detect_umbrae: bool = True,
    isolated_penumbra_threshold: float = 0.72
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Primary sunspot segmentation using morphological White Top-Hat transform.

    Parameters
    ----------
    norm_image : np.ndarray
        Limb-darkening normalized float image (quiet Sun ~ 1.0, sunspots < 1.0).
    disk_mask : np.ndarray
        Boolean solar disk mask.
    radius : int, default 21
        Radius of circular structuring element in pixels (at 1024x1024 scale).
    k_sigma : float, default 2.0
        Statistical multiplier for threshold: tau = mean + k_sigma * std.
    detect_umbrae : bool, default True
        Whether to also compute the sub-component umbral mask.
    isolated_penumbra_threshold : float, default 0.72
        Upper bound for adaptive umbra detection.

    Returns
    -------
    sunspot_mask : np.ndarray
        Binary uint8 mask of detected sunspot regions (penumbrae + umbrae).
    tophat_residual : np.ndarray
        Float array of the top-hat transformed inverted image.
    umbra_mask : np.ndarray or None
        Binary uint8 mask of detected umbral regions if detect_umbrae=True.
    """
    # Step 1: Invert image inside disk so sunspots become bright positive peaks
    # Quiet Sun (~1.0) maps to ~0.05, deep sunspots (~0.3) map to ~0.75
    inverted = np.zeros_like(norm_image, dtype=np.float32)
    max_ref = 1.05
    inverted[disk_mask] = np.maximum(0.0, max_ref - norm_image[disk_mask])

    # Step 2: Morphological White Top-Hat transform using circular structuring element
    se_size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
    tophat_residual = cv2.morphologyEx(inverted, cv2.MORPH_TOPHAT, kernel)

    # Step 3: Statistical cutoff threshold computed strictly on disk pixels
    disk_vals = tophat_residual[disk_mask]
    mean_val = float(np.mean(disk_vals))
    std_val = float(np.std(disk_vals))
    tau = mean_val + k_sigma * std_val

    # Step 4: Binary sunspot mask
    raw_sunspot = ((tophat_residual > tau) & disk_mask).astype(np.uint8)

    umbra_mask = None
    if detect_umbrae:
        umbra_mask, _ = decompose_umbra_penumbra(
            norm_image, raw_sunspot, umbra_threshold=isolated_penumbra_threshold
        )

    return raw_sunspot, tophat_residual, umbra_mask


def segment_otsu(
    norm_image: np.ndarray,
    disk_mask: np.ndarray
) -> np.ndarray:
    """
    Baseline 1: Global Otsu thresholding strictly on solar disk pixels.

    Parameters
    ----------
    norm_image : np.ndarray
        Limb-darkening normalized image.
    disk_mask : np.ndarray
        Boolean solar disk mask.

    Returns
    -------
    mask : np.ndarray
        Binary uint8 mask (1 = sunspot, 0 = background).
    """
    disk_pixels = norm_image[disk_mask]
    if len(disk_pixels) == 0:
        return np.zeros_like(norm_image, dtype=np.uint8)

    # Scale disk pixels to 8-bit [0, 255] for standard Otsu
    p_min, p_max = float(np.min(disk_pixels)), float(np.max(disk_pixels))
    if p_max <= p_min:
        return np.zeros_like(norm_image, dtype=np.uint8)

    norm_8u = np.clip((norm_image - p_min) / (p_max - p_min) * 255.0, 0, 255).astype(np.uint8)
    disk_8u = norm_8u[disk_mask]

    thresh_val = threshold_otsu(disk_8u)
    # Sunspots are dark, so intensities < thresh_val
    otsu_mask = ((norm_8u < thresh_val) & disk_mask).astype(np.uint8)
    return otsu_mask


def segment_adaptive(
    norm_image: np.ndarray,
    disk_mask: np.ndarray,
    block_size: int = 41,
    c_offset: float = 0.08
) -> np.ndarray:
    """
    Baseline 2: Local adaptive Gaussian thresholding within the solar disk.

    Parameters
    ----------
    norm_image : np.ndarray
        Limb-darkening normalized float image.
    disk_mask : np.ndarray
        Boolean solar disk mask.
    block_size : int, default 41
        Gaussian window size in pixels.
    c_offset : float, default 0.08
        Intensity contrast subtraction offset.

    Returns
    -------
    mask : np.ndarray
        Binary uint8 mask (1 = sunspot, 0 = background).
    """
    # Gaussian local mean
    local_mean = cv2.GaussianBlur(norm_image, (block_size, block_size), 0)

    # A pixel is candidate spot if it is darker than local mean by c_offset
    adaptive_mask = ((norm_image < (local_mean - c_offset)) & disk_mask).astype(np.uint8)
    return adaptive_mask
