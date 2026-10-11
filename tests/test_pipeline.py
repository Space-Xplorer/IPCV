"""
test_pipeline.py
----------------
Integration tests for end-to-end SunspotPipeline orchestrator.
"""

import os
import glob
import numpy as np
import pytest
from sunspot.pipeline import SunspotPipeline


@pytest.fixture
def sample_fits():
    daily_files = sorted(glob.glob("data/daily_fits/*.fits"))
    if not daily_files:
        pytest.skip("No daily FITS files found in data/daily_fits/")
    return daily_files[0]


def test_pipeline_single_image(sample_fits, tmp_path):
    cache_dir = str(tmp_path / "masks")
    pipe = SunspotPipeline(target_size=512, cache_dir=cache_dir, k_sigma=3.5, min_pore_area=8)

    # First run: compute and cache
    res = pipe.process_image(sample_fits, method="tophat", use_cache=True)

    assert "date" in res
    assert "num_spots" in res
    assert "total_area_msh" in res
    assert "features_df" in res
    assert "clean_mask" in res
    assert res["clean_mask"].shape == (512, 512)
    assert res["clean_mask"].dtype == np.uint8

    # Second run: must hit cache
    cached_npz = os.path.join(cache_dir, f"{res['date']}_tophat.npz")
    assert os.path.exists(cached_npz)

    res2 = pipe.process_image(sample_fits, method="tophat", use_cache=True)
    assert res2["num_spots"] == res["num_spots"]
    assert res2["total_area_msh"] == res["total_area_msh"]
