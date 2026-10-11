"""
features.py
-----------
Connected components labeling, spatial moments, and morphological feature extraction.

Implements extraction of:
1. Projected Solar Disk Area Fraction in parts-per-million (μDisk / ppm, commonly denoted A_f):
   A_f = (area_pixels / (pi * disk_radius^2)) * 10^6.
   (Uncorrected 2D projected disk area fraction; does not apply 1/cos(theta) spherical foreshortening).
2. Sub-pixel centroids and bounding boxes.
3. Shape descriptors (eccentricity from second-order central moments, solidity).
4. Photometric summary (mean and minimum core intensity).
5. Whole-disk aggregate activity index (Total Fractional Sunspot Area A_f).
6. Hierarchical umbra and penumbra morphology.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
import pandas as pd


def extract_features(
    clean_mask: np.ndarray,
    norm_image: np.ndarray,
    disk_radius: float,
    umbra_mask: Optional[np.ndarray] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extract individual sunspot region properties and disk-level aggregate metrics,
    including hierarchical umbra and penumbra morphology.

    Parameters
    ----------
    clean_mask : np.ndarray
        Cleaned binary uint8 sunspot mask (penumbra + umbra).
    norm_image : np.ndarray
        Limb-darkening normalized float image.
    disk_radius : float
        Radius of the solar disk in pixels.
    umbra_mask : np.ndarray, optional
        Binary uint8 mask of umbral cores. If None, umbra features are omitted.

    Returns
    -------
    features_df : pd.DataFrame
        Table with one row per detected sunspot component.
    summary : dict
        Disk-level aggregate statistics including total fractional area (A_f).
    """
    disk_area_px = np.pi * (disk_radius ** 2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        clean_mask, connectivity=8
    )

    records = []
    for i in range(1, num_labels):
        area_px = int(stats[i, cv2.CC_STAT_AREA])
        area_msh = (area_px / disk_area_px) * 1e6

        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[i][0]), float(centroids[i][1])

        # Region pixel slice for moments and photometry
        comp_mask = (labels == i)
        comp_intensities = norm_image[comp_mask]
        mean_int = float(np.mean(comp_intensities))
        min_int = float(np.min(comp_intensities))

        # Hierarchical Umbra vs Penumbra decomposition
        umbra_px = 0
        umbra_msh = 0.0
        penumbra_px = area_px
        penumbra_msh = area_msh
        up_ratio = 0.0
        if umbra_mask is not None:
            umbra_px = int(np.sum(umbra_mask[comp_mask]))
            umbra_msh = (umbra_px / disk_area_px) * 1e6
            penumbra_px = max(0, area_px - umbra_px)
            penumbra_msh = (penumbra_px / disk_area_px) * 1e6
            up_ratio = round(umbra_px / penumbra_px, 4) if penumbra_px > 0 else 0.0

        # Shape descriptors via image moments
        comp_uint8 = comp_mask.astype(np.uint8)
        moments = cv2.moments(comp_uint8)

        # Eccentricity from eigenvalues of second central moments
        mu20 = moments["mu20"]
        mu02 = moments["mu02"]
        mu11 = moments["mu11"]
        delta = np.sqrt((mu20 - mu02) ** 2 + 4 * (mu11 ** 2))
        lambda1 = (mu20 + mu02 + delta) / 2.0
        lambda2 = (mu20 + mu02 - delta) / 2.0

        if lambda1 > 0 and lambda2 >= 0:
            eccentricity = float(np.sqrt(np.clip(1.0 - lambda2 / lambda1, 0.0, 1.0)))
        else:
            eccentricity = 0.0

        # Solidity (area / convex hull area)
        contours, _ = cv2.findContours(comp_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            hull = cv2.convexHull(contours[0])
            hull_area = cv2.contourArea(hull)
            solidity = float(np.clip(area_px / hull_area, 0.0, 1.0)) if hull_area > 0 else 1.0
        else:
            solidity = 1.0

        eq_radius_px = float(np.sqrt(area_px / np.pi))

        rec = {
            "component_id": i,
            "centroid_x": round(cx, 2),
            "centroid_y": round(cy, 2),
            "area_pixels": area_px,
            "area_msh": round(area_msh, 3),
            "equivalent_radius_px": round(eq_radius_px, 2),
            "bbox_x": bx,
            "bbox_y": by,
            "bbox_w": bw,
            "bbox_h": bh,
            "eccentricity": round(eccentricity, 4),
            "solidity": round(solidity, 4),
            "mean_intensity": round(mean_int, 4),
            "min_intensity": round(min_int, 4),
        }
        if umbra_mask is not None:
            rec["umbra_pixels"] = umbra_px
            rec["umbra_msh"] = round(umbra_msh, 3)
            rec["penumbra_pixels"] = penumbra_px
            rec["penumbra_msh"] = round(penumbra_msh, 3)
            rec["up_ratio"] = up_ratio

        records.append(rec)

    features_df = pd.DataFrame(records)

    # Disk-level aggregate activity metrics
    total_area_px = int(np.sum(clean_mask))
    total_area_msh = (total_area_px / disk_area_px) * 1e6
    num_spots = len(records)

    summary = {
        "num_spots": num_spots,
        "total_area_pixels": total_area_px,
        "total_area_msh": round(float(total_area_msh), 2),
        "mean_spot_area_msh": round(float(features_df["area_msh"].mean()), 2) if num_spots > 0 else 0.0,
        "max_spot_area_msh": round(float(features_df["area_msh"].max()), 2) if num_spots > 0 else 0.0,
    }

    if umbra_mask is not None:
        total_u_px = int(np.sum(umbra_mask[clean_mask > 0]))
        total_u_msh = (total_u_px / disk_area_px) * 1e6
        total_p_msh = max(0.0, total_area_msh - total_u_msh)
        summary["total_umbra_msh"] = round(float(total_u_msh), 2)
        summary["total_penumbra_msh"] = round(float(total_p_msh), 2)
        summary["overall_up_ratio"] = round(float(total_u_msh / total_p_msh), 4) if total_p_msh > 0 else 0.0

    return features_df, summary
