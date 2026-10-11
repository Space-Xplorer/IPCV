"""
pipeline.py
-----------
End-to-end solar computer vision pipeline orchestrator with caching.

Coordinates:
FITS Ingestion -> Geometry & Disk Masking -> 1024x1024 Downsampling ->
Limb-Darkening Correction -> Segmentation (Top-Hat / Otsu / Adaptive) ->
Morphological Cleanup -> Region Feature Extraction -> Daily Aggregation.
"""

from typing import Dict, Any, List, Optional, Union
import os
import re
import glob
import time
import numpy as np
import pandas as pd

from sunspot.disk import (
    read_fits_data_and_geometry,
    scale_geometry,
    create_disk_mask
)
from sunspot.normalize import (
    downsample_image,
    correct_limb_darkening
)
from sunspot.segment import (
    segment_tophat,
    segment_otsu,
    segment_adaptive,
    decompose_umbra_penumbra
)
from sunspot.cleanup import cleanup_mask
from sunspot.features import extract_features


FNAME_DATE_RE = re.compile(
    r"hmi\.ic_45s\.(\d{4})\.(\d{2})\.(\d{2})"
)


class SunspotPipeline:
    """
    Classical vision orchestrator for solar continuum images.
    """

    def __init__(
        self,
        target_size: int = 1024,
        cache_dir: Optional[str] = "results/masks",
        tophat_radius: int = 21,
        k_sigma: float = 3.5,
        min_pore_area: int = 8
    ):
        self.target_size = target_size
        self.cache_dir = cache_dir
        self.tophat_radius = tophat_radius
        self.k_sigma = k_sigma
        self.min_pore_area = min_pore_area

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def extract_date_from_path(self, fits_path: str) -> str:
        """Extract ISO YYYY-MM-DD date string from filename."""
        fname = os.path.basename(fits_path)
        m = FNAME_DATE_RE.search(fname)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        return "UNKNOWN"

    def process_image(
        self,
        fits_path: str,
        method: str = "tophat",
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Execute full pipeline on a single SDO/HMI continuum FITS file.

        Parameters
        ----------
        fits_path : str
            Path to FITS file.
        method : str, default 'tophat'
            Segmentation method ('tophat', 'otsu', or 'adaptive').
        use_cache : bool, default True
            Whether to read/write cached segmentation masks.

        Returns
        -------
        result : dict
            Contains 'date', 'num_spots', 'total_area_msh', 'features_df',
            'clean_mask', 'norm_image', 'disk_mask', 'scaled_geometry'.
        """
        date_str = self.extract_date_from_path(fits_path)
        cache_file = None
        if self.cache_dir and use_cache and date_str != "UNKNOWN":
            cache_file = os.path.join(self.cache_dir, f"{date_str}_{method}.npz")

        # Step 1: Read raw FITS
        data, meta, raw_geom = read_fits_data_and_geometry(fits_path)

        # Step 2: Downsample to 1024x1024
        small_img, scale_factor = downsample_image(data, self.target_size)
        scaled_geom = scale_geometry(raw_geom, scale_factor)
        cx, cy, r_scaled = scaled_geom

        # Step 3: Circular Disk Mask
        disk_mask = create_disk_mask(small_img.shape, (cx, cy), r_scaled, r_fraction=0.92)

        # Step 4: Limb-Darkening Normalization
        norm_img, poly_fit = correct_limb_darkening(
            small_img, disk_mask, (cx, cy), r_scaled, apply_median_filter=True
        )

        # Step 5: Check cache for mask
        clean_mask = None
        if cache_file and os.path.exists(cache_file):
            try:
                npz = np.load(cache_file)
                clean_mask = npz["clean_mask"]
            except Exception:
                clean_mask = None

        if clean_mask is None:
            # Execute segmentation
            if method == "tophat":
                raw_mask, _, _ = segment_tophat(
                    norm_img, disk_mask,
                    radius=self.tophat_radius,
                    k_sigma=self.k_sigma
                )
            elif method == "otsu":
                raw_mask = segment_otsu(norm_img, disk_mask)
            elif method == "adaptive":
                raw_mask = segment_adaptive(norm_img, disk_mask)
            else:
                raise ValueError(f"Unknown segmentation method: {method}")

            # Morphological cleanup
            clean_mask = cleanup_mask(raw_mask, min_area_pixels=self.min_pore_area)

            # Save to cache
            if cache_file:
                np.savez_compressed(cache_file, clean_mask=clean_mask)

        # Step 6: Hierarchical Umbra/Penumbra Decomposition
        umbra_mask, penumbra_mask = decompose_umbra_penumbra(norm_img, clean_mask)

        # Step 7: Extract Region Features
        features_df, summary = extract_features(
            clean_mask, norm_img, r_scaled, umbra_mask=umbra_mask
        )

        return {
            "date": date_str,
            "fits_path": fits_path,
            "method": method,
            "num_spots": summary["num_spots"],
            "total_area_pixels": summary["total_area_pixels"],
            "total_area_msh": summary["total_area_msh"],
            "total_umbra_msh": summary.get("total_umbra_msh", 0.0),
            "total_penumbra_msh": summary.get("total_penumbra_msh", 0.0),
            "overall_up_ratio": summary.get("overall_up_ratio", 0.0),
            "mean_spot_area_msh": summary["mean_spot_area_msh"],
            "max_spot_area_msh": summary["max_spot_area_msh"],
            "features_df": features_df,
            "clean_mask": clean_mask,
            "umbra_mask": umbra_mask,
            "penumbra_mask": penumbra_mask,
            "norm_image": norm_img,
            "disk_mask": disk_mask,
            "scaled_geometry": scaled_geom,
            "raw_image": small_img,
        }

    def process_batch(
        self,
        fits_input: Union[str, List[str]],
        method: str = "tophat",
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Batch process a directory or list of FITS files and return daily activity DataFrame.

        Parameters
        ----------
        fits_input : str or list of str
            Directory containing FITS files or explicit list of FITS paths.
        method : str, default 'tophat'
        verbose : bool, default True

        Returns
        -------
        daily_df : pd.DataFrame
            Sorted DataFrame with daily measurements.
        """
        if isinstance(fits_input, str):
            if os.path.isdir(fits_input):
                fits_files = sorted(glob.glob(os.path.join(fits_input, "*.fits")))
            else:
                fits_files = sorted(glob.glob(fits_input))
        else:
            fits_files = sorted(fits_input)

        if not fits_files:
            raise FileNotFoundError(f"No FITS files found in {fits_input}")

        t0 = time.time()
        records = []
        if verbose:
            print(f"Batch processing {len(fits_files)} daily images using '{method}'...")

        for idx, fpath in enumerate(fits_files, 1):
            res = self.process_image(fpath, method=method)
            records.append({
                "date": res["date"],
                "fits_filename": os.path.basename(fpath),
                "method": method,
                "num_spots": res["num_spots"],
                "total_area_pixels": res["total_area_pixels"],
                "total_area_msh": res["total_area_msh"],
                "total_umbra_msh": res["total_umbra_msh"],
                "total_penumbra_msh": res["total_penumbra_msh"],
                "overall_up_ratio": res["overall_up_ratio"],
                "mean_spot_area_msh": res["mean_spot_area_msh"],
                "max_spot_area_msh": res["max_spot_area_msh"],
            })
            if verbose and (idx % 15 == 0 or idx == len(fits_files)):
                elapsed = time.time() - t0
                print(f"  [{idx:2d}/{len(fits_files)}] {res['date']} - spots: {res['num_spots']:2d}, area: {res['total_area_msh']:6.1f} MSH ({elapsed:.1f}s)")

        daily_df = pd.DataFrame(records)
        daily_df.sort_values("date", inplace=True)
        daily_df.reset_index(drop=True, inplace=True)
        return daily_df
