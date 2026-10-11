"""
test_activity.py
----------------
Unit tests for SILSO parsing and activity metric correlation.
"""

import numpy as np
import pandas as pd
import pytest
from sunspot.activity import correlate_activity_metrics


def test_correlate_activity_metrics_linear():
    # Synthetic perfectly correlated series
    dates = pd.date_range("2023-10-01", periods=10).strftime("%Y-%m-%d")
    pipeline_df = pd.DataFrame({
        "date": dates,
        "total_area_msh": np.linspace(500, 2500, 10),
        "num_spots": np.linspace(10, 50, 10),
    })

    silso_df = pd.DataFrame({
        "date": dates,
        "silso_sn": np.linspace(40, 160, 10),
    })

    merged, metrics = correlate_activity_metrics(pipeline_df, silso_df)
    assert len(merged) == 10
    assert np.isclose(metrics["pearson_r"], 1.0, atol=1e-3)
    assert np.isclose(metrics["spearman_rho"], 1.0, atol=1e-3)
    assert np.isclose(metrics["r_squared"], 1.0, atol=1e-3)


def test_correlate_activity_metrics_missing_dates():
    dates1 = ["2023-10-01", "2023-10-02", "2023-10-03"]
    dates2 = ["2023-10-02", "2023-10-03", "2023-10-04"]  # disjoint dates
    pipeline_df = pd.DataFrame({"date": dates1, "total_area_msh": [100.0, 200.0, 300.0], "num_spots": [1, 2, 3]})
    silso_df = pd.DataFrame({"date": dates2, "silso_sn": [50.0, 75.0, 100.0]})

    merged, metrics = correlate_activity_metrics(pipeline_df, silso_df)
    assert len(merged) == 2
    assert list(merged["date"]) == ["2023-10-02", "2023-10-03"]

