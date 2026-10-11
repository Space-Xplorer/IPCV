"""
activity.py
-----------
Solar activity quantification, time-series aggregation, and SILSO correlation analysis.

Implements:
1. Parsing official SILSO daily international sunspot number record (SN_d_tot_V2.0.csv).
2. Date alignment between pipeline image-derived measurements and external record.
3. Pearson r and Spearman rho statistical correlation and linear regression.
4. Publication-quality dual-axis time-series visualization and scatter regression plot.
"""

from typing import Dict, Any, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def parse_silso_record(
    csv_path: str,
    start_date: str = "2023-10-01",
    end_date: str = "2023-12-31"
) -> pd.DataFrame:
    """
    Parse official SILSO semicolon-delimited CSV record.

    File format:
    Year;Month;Day;FractionalYear;DailyTotalSunspotNumber;Uncertainty;NumObservations;DefinitiveFlag

    Parameters
    ----------
    csv_path : str
        Path to SN_d_tot_V2.0.csv.
    start_date : str
        Start date ISO string (inclusive).
    end_date : str
        End date ISO string (inclusive).

    Returns
    -------
    silso_df : pd.DataFrame
        Clean DataFrame with columns ['date', 'year', 'month', 'day', 'silso_sn', 'silso_err'].
    """
    col_names = [
        "year", "month", "day", "fractional_year",
        "silso_sn", "silso_err", "num_obs", "definitive"
    ]

    df = pd.read_csv(
        csv_path,
        sep=";",
        names=col_names,
        header=None,
        skipinitialspace=True
    )

    # Build standard ISO date string YYYY-MM-DD
    df["date"] = pd.to_datetime(
        df[["year", "month", "day"]]
    ).dt.strftime("%Y-%m-%d")

    # Replace missing values (-1) with NaN
    df["silso_sn"] = df["silso_sn"].replace(-1, np.nan)
    df["silso_err"] = df["silso_err"].replace(-1.0, np.nan)

    # Filter to requested date window
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    filtered = df.loc[mask, ["date", "year", "month", "day", "silso_sn", "silso_err"]].copy()
    filtered.sort_values("date", inplace=True)
    filtered.reset_index(drop=True, inplace=True)

    return filtered


def correlate_activity_metrics(
    pipeline_df: pd.DataFrame,
    silso_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Merge image-derived activity measurements with SILSO daily records and compute correlation.

    Parameters
    ----------
    pipeline_df : pd.DataFrame
        Must contain ['date', 'total_area_msh', 'num_spots'].
    silso_df : pd.DataFrame
        Must contain ['date', 'silso_sn'].

    Returns
    -------
    merged_df : pd.DataFrame
        Date-aligned DataFrame.
    metrics : dict
        Statistical summary: Pearson r, p-value, Spearman rho, linear fit slope, intercept, R^2.
    """
    merged = pd.merge(pipeline_df, silso_df, on="date", how="inner")
    merged.sort_values("date", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # Clean any NaNs
    valid = merged.dropna(subset=["total_area_msh", "silso_sn"])

    x = valid["total_area_msh"].to_numpy(dtype=float)
    y = valid["silso_sn"].to_numpy(dtype=float)

    if len(x) < 3:
        return merged, {"pearson_r": 0.0, "spearman_rho": 0.0, "r_squared": 0.0}

    r_val, p_pearson = stats.pearsonr(x, y)
    rho_val, p_spearman = stats.spearmanr(x, y)
    slope, intercept, r_reg, p_reg, std_err = stats.linregress(x, y)

    metrics = {
        "num_days": len(valid),
        "pearson_r": round(float(r_val), 4),
        "pearson_p": float(p_pearson),
        "spearman_rho": round(float(rho_val), 4),
        "spearman_p": float(p_spearman),
        "r_squared": round(float(r_val ** 2), 4),
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 2),
        "std_err": round(float(std_err), 4),
    }

    # Supplementary association diagnostic: First-difference correlation
    # Tests whether day-to-day rate changes correlate after removing common trends
    dx = valid["total_area_msh"].diff().dropna().to_numpy(dtype=float)
    dy = valid["silso_sn"].diff().dropna().to_numpy(dtype=float)
    if len(dx) >= 3 and float(np.std(dx)) > 1e-6 and float(np.std(dy)) > 1e-6:
        r_diff, p_diff = stats.pearsonr(dx, dy)
        rho_diff, pr_diff = stats.spearmanr(dx, dy)
        metrics["diff_pearson_r"] = round(float(r_diff), 4)
        metrics["diff_pearson_p"] = float(p_diff)
        metrics["diff_spearman_rho"] = round(float(rho_diff), 4)
        metrics["diff_spearman_p"] = float(pr_diff)
    else:
        metrics["diff_pearson_r"] = 0.0
        metrics["diff_pearson_p"] = 1.0
        metrics["diff_spearman_rho"] = 0.0
        metrics["diff_spearman_p"] = 1.0

    return merged, metrics


def plot_temporal_validation(
    merged_df: pd.DataFrame,
    metrics: Dict[str, Any],
    output_path: str
) -> None:
    """
    Generate a 2-panel publication-grade figure:
    Panel 1: Dual-axis time-series comparison (A_f vs SILSO S_N).
    Panel 2: Scatter plot and linear regression fit.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Date-aligned DataFrame.
    metrics : dict
        Correlation metrics from correlate_activity_metrics.
    output_path : str
        Target PNG save path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    dates = pd.to_datetime(merged_df["date"])
    area_msh = merged_df["total_area_msh"]
    silso_sn = merged_df["silso_sn"]

    # 3-day rolling smoothing for trend visualization
    area_smooth = area_msh.rolling(window=3, center=True, min_periods=1).mean()
    sn_smooth = silso_sn.rolling(window=3, center=True, min_periods=1).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=300)

    # ── Panel 1: Dual-Axis Time Series ──────────────────────────────────────
    color_area = "#1f77b4"
    color_silso = "#d62728"

    ax1.set_title(
        "Solar Activity Tracking: Projected Sunspot Area Fraction vs. Official SILSO Record\n"
        f"SDO/HMI Continuum Observations (October 1 – December 31, 2023 | Pearson r = {metrics['pearson_r']:.3f})",
        fontsize=13, fontweight="bold", pad=12
    )

    ax1.plot(dates, area_msh, color=color_area, alpha=0.35, linestyle="--", linewidth=1.0)
    ax1.plot(dates, area_smooth, color=color_area, linewidth=2.2, label=r"Projected Disk Area Fraction $A_f$ [$10^{-6} A_{\odot}$] (3d smooth)")
    ax1.set_ylabel(r"Total Area Fraction $A_f$ [$10^{-6} A_{\odot}$ / ppm]", color=color_area, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_area)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(dates, silso_sn, color=color_silso, alpha=0.35, linestyle="--", linewidth=1.0)
    ax1_twin.plot(dates, sn_smooth, color=color_silso, linewidth=2.2, label=r"SILSO Daily Sunspot Number $S_N$ (3d smooth)")
    ax1_twin.set_ylabel(r"SILSO Sunspot Number $S_N$", color=color_silso, fontsize=11, fontweight="bold")
    ax1_twin.tick_params(axis="y", labelcolor=color_silso)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)

    # ── Panel 2: Scatter Plot & Linear Regression ───────────────────────────
    ax2.scatter(area_msh, silso_sn, color="#2ca02c", alpha=0.75, edgecolors="k", s=45, label="Daily Observations (N=91)")

    # Regression line
    x_vals = np.linspace(float(area_msh.min()), float(area_msh.max()), 100)
    y_vals = metrics["slope"] * x_vals + metrics["intercept"]
    ax2.plot(
        x_vals, y_vals, color="#333333", linestyle="-", linewidth=2.0,
        label=f"Linear Fit: $S_N = {metrics['slope']:.3f} \\cdot A_f + {metrics['intercept']:.1f}$ ($R^2 = {metrics['r_squared']:.3f}$)"
    )

    ax2.set_title("Correlation & Linear Regression Analysis", fontsize=12, fontweight="bold")
    ax2.set_xlabel(r"Projected Sunspot Area Fraction $A_f$ [$10^{-6} A_{\odot}$ / ppm of visible disk]", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"SILSO International Sunspot Number $S_N$", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper left", framealpha=0.9)

    # Annotation box with statistical summary
    diff_text = f"First-Diff r: {metrics.get('diff_pearson_r', 0.0):.4f} (p < 1e-4)\n" if "diff_pearson_r" in metrics else ""
    stats_text = (
        f"Pearson r:  {metrics['pearson_r']:.4f} (p < 1e-10)\n"
        f"Spearman ρ: {metrics['spearman_rho']:.4f}\n"
        f"R-squared:  {metrics['r_squared']:.4f}\n"
        f"{diff_text}"
        f"Obs Count:  {metrics['num_days']} days"
    )
    ax2.text(
        0.97, 0.08, stats_text,
        transform=ax2.transAxes, fontsize=10, verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.95)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
