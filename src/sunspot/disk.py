"""
disk.py
-------
Solar disk geometry detection, astrometric WCS parsing, and mask generation.

Implements:
1. Authoritative FITS WCS header extraction (CRPIX1, CRPIX2, RSUN_OBS, CDELT1)
   derived from Reference 1 (SDO/HFC production pipeline).
2. Analytical circular disk mask generation with inner limb buffer (r_fraction=0.92,
   retaining 84.64% of disk area, excluding outermost 15.36% annulus).
3. Educational classical disk detector (thresholding + contour enclosing circle)
   for IPCV syllabus Unit IV alignment and ablation.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
from astropy.io import fits


def read_fits_data_and_geometry(
    fits_path: str
) -> Tuple[np.ndarray, Dict[str, Any], Tuple[float, float, float]]:
    """
    Read full-disk solar image and astrometric geometry from SDO/HMI FITS file.

    Parameters
    ----------
    fits_path : str
        Path to SDO/HMI continuum FITS file.

    Returns
    -------
    image : np.ndarray
        2D float array containing solar continuum intensities (with off-limb NaNs).
    meta : dict
        Extracted observation metadata (T_REC, DATE-OBS, INSTRUME, QUALITY, etc.).
    geometry : tuple (cx, cy, radius_px)
        Solar center x, center y (0-indexed pixel coordinates), and solar radius in pixels.
    """
    with fits.open(fits_path) as hdul:
        # SDO/HMI FITS files often store data in extension 1 or extension 0
        if len(hdul) > 1 and hdul[1].data is not None:
            header = hdul[1].header
            data = hdul[1].data.astype(np.float32)
        else:
            header = hdul[0].header
            data = hdul[0].data.astype(np.float32)

    # Check if WCS keywords are actually present in FITS header
    has_wcs = ("CRPIX1" in header) and (header.get("CRPIX1") is not None)

    if has_wcs:
        meta = {
            "t_rec": str(header.get("T_REC", header.get("DATE-OBS", "UNKNOWN"))),
            "date_obs": str(header.get("DATE-OBS", "UNKNOWN")),
            "instrument": str(header.get("INSTRUME", "HMI")),
            "quality": int(header.get("QUALITY", 0)),
            "crpix1": float(header["CRPIX1"]),
            "crpix2": float(header["CRPIX2"]),
            "rsun_obs": float(header.get("RSUN_OBS", 959.0)),
            "cdelt1": float(header.get("CDELT1", 0.504)),
        }
        cx = meta["crpix1"] - 1.0
        cy = meta["crpix2"] - 1.0
        cdelt = meta["cdelt1"] if meta["cdelt1"] > 0 else 0.504
        radius_px = meta["rsun_obs"] / cdelt
    else:
        # Fallback to classical computer vision disk detection
        cx, cy, radius_px = detect_disk_classical(data)
        meta = {
            "t_rec": str(header.get("T_REC", header.get("DATE-OBS", "UNKNOWN"))),
            "date_obs": str(header.get("DATE-OBS", "UNKNOWN")),
            "instrument": str(header.get("INSTRUME", "HMI")),
            "quality": int(header.get("QUALITY", 0)),
            "crpix1": float(cx + 1.0),
            "crpix2": float(cy + 1.0),
            "rsun_obs": 959.0,
            "cdelt1": 0.504,
        }

    return data, meta, (cx, cy, radius_px)


def create_disk_mask(
    shape: Tuple[int, int],
    center: Tuple[float, float],
    radius: float,
    r_fraction: float = 0.92
) -> np.ndarray:
    """
    Generate boolean circular mask for the visible solar disk.

    Parameters
    ----------
    shape : tuple (height, width)
        Target 2D array shape.
    center : tuple (cx, cy)
        Disk center coordinates (x=column, y=row).
    radius : float
        Solar radius in pixels.
    r_fraction : float, default 0.92
        Radial fractional cutoff. Setting r_fraction=0.92 retains (0.92)^2 = 84.64%
        of the visible disk area, excluding the outermost 15.36% annulus (mu = cos(theta) < 0.392).
        This eliminates steep limb-darkening gradient divergence and extreme foreshortening,
        with the scientific trade-off that sunspots within ~1-2 days of limb entry/exit are excluded.

    Returns
    -------
    mask : np.ndarray
        Boolean 2D mask (True inside solar disk, False outside).
    """
    height, width = shape
    y_coords, x_coords = np.ogrid[:height, :width]
    cx, cy = center
    dist_sq = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
    r_cutoff_sq = (radius * r_fraction) ** 2
    return dist_sq <= r_cutoff_sq


def scale_geometry(
    geometry: Tuple[float, float, float],
    scale_factor: float
) -> Tuple[float, float, float]:
    """
    Scale center coordinates and radius by a spatial downsampling factor.

    Parameters
    ----------
    geometry : tuple (cx, cy, radius)
    scale_factor : float

    Returns
    -------
    scaled_geometry : tuple (cx * s, cy * s, radius * s)
    """
    cx, cy, r = geometry
    return (cx * scale_factor, cy * scale_factor, r * scale_factor)


def detect_disk_classical(
    image: np.ndarray,
    threshold_fraction: float = 0.10
) -> Tuple[float, float, float]:
    """
    Educational classical solar disk detector for IPCV syllabus Unit IV alignment.
    Detects solar disk boundary using thresholding and contour minimum enclosing circle.

    Parameters
    ----------
    image : np.ndarray
        2D intensity image (can contain NaNs outside disk).
    threshold_fraction : float
        Fraction of maximum valid intensity used to segment the solar disk from background.

    Returns
    -------
    geometry : tuple (cx, cy, radius_px)
    """
    # Replace NaNs with zero and normalize to 8-bit
    valid = np.nan_to_num(image, nan=0.0)
    max_val = np.max(valid)
    if max_val <= 0:
        h, w = image.shape
        return (w / 2.0, h / 2.0, min(h, w) * 0.45)

    thresh = (valid > (max_val * threshold_fraction)).astype(np.uint8) * 255

    # Morphological closing to fill any internal holes or sunspot voids
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find the largest outer contour
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = image.shape
        return (w / 2.0, h / 2.0, min(h, w) * 0.45)

    largest_contour = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
    return (float(cx), float(cy), float(radius))
