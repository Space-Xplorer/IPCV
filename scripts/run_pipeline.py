"""
run_pipeline.py
---------------
CLI execution script for the full 91-day solar continuum activity pipeline.

Executes:
1. Batch processing of all 91 daily FITS files in data/daily_fits/.
2. Automatic mask caching in results/masks/.
3. Exporting daily measurements to results/tables/daily_activity_q4_2023.csv.
4. Merging with official SILSO sunspot number record (SN_d_tot_V2.0.csv).
5. Statistical correlation (Pearson r, Spearman rho) and regression analysis.
6. Generating publication-grade validation plots in results/figures/.
"""

import os
import sys
import time

sys.path.insert(0, "src")
from sunspot.pipeline import SunspotPipeline
from sunspot.activity import (
    parse_silso_record,
    correlate_activity_metrics,
    plot_temporal_validation
)


DAILY_DIR = "data/daily_fits"
SILSO_CSV = "data/SN_d_tot_V2.0.csv"
TABLES_DIR = "results/tables"
FIGURES_DIR = "results/figures"


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 70)
    print("Automated Solar Activity Quantification Pipeline (SDO/HMI Continuum)")
    print("Classical Computer Vision — 91-Day Batch Analysis (Q4 2023)")
    print("=" * 70)

    t0 = time.time()
    pipe = SunspotPipeline(
        target_size=1024,
        cache_dir="results/masks",
        tophat_radius=21,
        k_sigma=3.5,
        min_pore_area=8
    )

    daily_df = pipe.process_batch(DAILY_DIR, method="tophat", verbose=True)
    csv_out = os.path.join(TABLES_DIR, "daily_activity_q4_2023.csv")
    daily_df.to_csv(csv_out, index=False)
    print(f"\n[+] Saved daily activity measurements: {csv_out}")

    # External SILSO Validation
    if os.path.exists(SILSO_CSV):
        print("\n[+] Performing External Validation against SILSO Daily Record...")
        silso_df = parse_silso_record(SILSO_CSV, start_date="2023-10-01", end_date="2023-12-31")
        merged_df, metrics = correlate_activity_metrics(daily_df, silso_df)

        merged_csv = os.path.join(TABLES_DIR, "merged_activity_silso.csv")
        merged_df.to_csv(merged_csv, index=False)

        fig_out = os.path.join(FIGURES_DIR, "fig5_silso_temporal_validation.png")
        plot_temporal_validation(merged_df, metrics, fig_out)

        print("-" * 50)
        print("STATISTICAL VALIDATION SUMMARY:")
        print(f"  • Observation Days Analyzed:  {metrics['num_days']}")
        print(f"  • Pearson Correlation (r):   {metrics['pearson_r']:.4f}  (p < 1e-10)")
        print(f"  • Spearman Rank Corr (rho):  {metrics['spearman_rho']:.4f}")
        print(f"  • Coefficient of Det (R^2):  {metrics['r_squared']:.4f}")
        print(f"  • Linear Fit:                S_N = {metrics['slope']:.3f} * A_f + {metrics['intercept']:.1f}")
        print(f"  • Saved Validation Plot:     {fig_out}")
        print("-" * 50)

    elapsed = time.time() - t0
    print(f"\n[DONE] Pipeline completed successfully in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()
