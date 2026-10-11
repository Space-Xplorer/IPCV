"""
app.py
------
Interactive Streamlit Dashboard for:
Automated Solar Activity Quantification Using Classical Computer Vision

Run with:
    streamlit run app.py
"""

import os
import sys
import glob
import re

# Ensure src/ is on Python search path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

from sunspot.disk import (
    read_fits_data_and_geometry,
    scale_geometry,
    create_disk_mask,
)
from sunspot.normalize import downsample_image, correct_limb_darkening
from sunspot.segment import (
    segment_tophat,
    segment_otsu,
    segment_adaptive,
    decompose_umbra_penumbra,
)
from sunspot.cleanup import cleanup_mask
from sunspot.features import extract_features
from sunspot.activity import (
    parse_silso_record,
    correlate_activity_metrics,
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Solar Activity Classical CV Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffaa00;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #b0bec5;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 12px 18px;
        border-left: 4px solid #ffaa00;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Helper Functions & Caching
# ---------------------------------------------------------
@st.cache_data
def get_fits_file_list(fits_dir: str = "data/daily_fits"):
    files = sorted(glob.glob(os.path.join(fits_dir, "*.fits")))
    file_map = {}
    date_re = re.compile(r"hmi\.ic_45s\.(\d{4})\.(\d{2})\.(\d{2})")
    for f in files:
        m = date_re.search(os.path.basename(f))
        if m:
            date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            file_map[date_str] = f
    return file_map


@st.cache_data
def load_temporal_results():
    csv_path = "results/tables/daily_activity_q4_2023.csv"
    silso_path = "data/SN_d_tot_V2.0.csv"
    if os.path.exists(csv_path) and os.path.exists(silso_path):
        daily_df = pd.read_csv(csv_path)
        silso_df = parse_silso_record(silso_path)
        merged, metrics = correlate_activity_metrics(daily_df, silso_df)
        return daily_df, silso_df, merged, metrics
    return None, None, None, None


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/b/b4/The_Sun_by_the_Atmospheric_Imaging_Assembly_of_NASA%27s_Solar_Dynamics_Observatory_-_20100819.jpg", use_container_width=True)
st.sidebar.markdown("### ☀️ SDO/HMI Solar Inspector")

fits_map = get_fits_file_list()
dates = sorted(list(fits_map.keys()))

if not dates:
    st.error("No FITS files found in `data/daily_fits/`. Please check data directory.")
    st.stop()

selected_date = st.sidebar.selectbox("Observation Date", options=dates, index=len(dates)//2)
fits_file = fits_map[selected_date]

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Segmentation Parameters")

method_choice = st.sidebar.selectbox(
    "Algorithm Paradigm",
    options=["White Top-Hat (Morphological)", "Global Otsu Thresholding", "Local Adaptive Gaussian"],
    index=0
)

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    tophat_r = st.slider("Top-Hat Radius (px)", min_value=7, max_value=35, value=21, step=2)
    min_pore_area = st.slider("Min Area (px)", min_value=2, max_value=25, value=8, step=1)
with col_s2:
    k_sigma = st.slider("Multiplier k_sigma", min_value=1.5, max_value=5.0, value=3.5, step=0.1)
    umbra_thresh = st.slider("Umbra Cutoff", min_value=0.50, max_value=0.85, value=0.72, step=0.01)

st.sidebar.markdown("---")
st.sidebar.caption("Classical Computer Vision Course Project (IPCV) — 100% Non-Deep-Learning")


# ---------------------------------------------------------
# Core Processing
# ---------------------------------------------------------
@st.cache_data
def process_single_frame(fpath, method_key, r_th, k_sig, min_area, u_thresh):
    # Read and downsample
    raw_data, meta, raw_geom = read_fits_data_and_geometry(fpath)
    small_img, scale_factor = downsample_image(raw_data, target_size=1024)
    scaled_geom = scale_geometry(raw_geom, scale_factor)
    cx, cy, r_scaled = scaled_geom

    # Disk mask & Limb darkening
    disk_mask = create_disk_mask(small_img.shape, (cx, cy), r_scaled, r_fraction=0.92)
    norm_img, poly_fit = correct_limb_darkening(
        small_img, disk_mask, (cx, cy), r_scaled, apply_median_filter=True
    )

    # Segment
    if "Top-Hat" in method_key:
        raw_mask, residual, _ = segment_tophat(
            norm_img, disk_mask, radius=r_th, k_sigma=k_sig
        )
    elif "Otsu" in method_key:
        raw_mask = segment_otsu(norm_img, disk_mask)
        residual = None
    else:
        raw_mask = segment_adaptive(norm_img, disk_mask)
        residual = None

    clean_mask = cleanup_mask(raw_mask, min_area_pixels=min_area)
    umbra_mask, penumbra_mask = decompose_umbra_penumbra(
        norm_img, clean_mask, umbra_threshold=u_thresh
    )

    features_df, summary = extract_features(
        clean_mask, norm_img, r_scaled, umbra_mask=umbra_mask
    )

    return (
        small_img, norm_img, disk_mask, clean_mask,
        umbra_mask, penumbra_mask, features_df, summary, (cx, cy, r_scaled)
    )

(
    raw_img, norm_img, disk_mask, clean_mask,
    umbra_mask, penumbra_mask, features_df, summary, geom
) = process_single_frame(fits_file, method_choice, tophat_r, k_sigma, min_pore_area, umbra_thresh)

# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.markdown('<p class="main-header">Automated Solar Activity Quantification</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Classical Computer Vision Pipeline for SDO/HMI Continuum Imagery, '
    'Hierarchical Umbra/Penumbra Decomposition, and SILSO International Sunspot Validation</p>',
    unsafe_allow_html=True
)

# Metric KPI Bar
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("Total Spots Detected", f"{summary['num_spots']}")
with kpi2:
    st.metric("Total Area (A_f)", f"{summary['total_area_msh']:.1f} ppm / µDisk")
with kpi3:
    st.metric("Umbra Area", f"{summary.get('total_umbra_msh', 0.0):.1f} ppm")
with kpi4:
    st.metric("Penumbra Area", f"{summary.get('total_penumbra_msh', 0.0):.1f} ppm")
with kpi5:
    st.metric("Umbra/Penumbra Ratio", f"{summary.get('overall_up_ratio', 0.0):.3f}")

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Hierarchical Segmentation & Overlays",
    "📊 Morphological Feature Analysis",
    "📈 91-Day Temporal Tracking & SILSO",
    "🔬 Benchmark Experiments & Syllabus"
])

# ---------------------------------------------------------
# Tab 1: Segmentation & Overlays
# ---------------------------------------------------------
with tab1:
    st.markdown(f"#### Observation: `{selected_date}` | Method: `{method_choice}`")
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.caption("1. Raw SDO/HMI Continuum")
        fig1, ax1 = plt.subplots(figsize=(4, 4), facecolor="#0e1117")
        ax1.imshow(raw_img, cmap="copper", origin="lower")
        ax1.axis("off")
        fig1.tight_layout(pad=0)
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        st.caption("2. 5th-Order Limb-Corrected (I_norm)")
        fig2, ax2 = plt.subplots(figsize=(4, 4), facecolor="#0e1117")
        ax2.imshow(norm_img, cmap="afmhot", vmin=0.4, vmax=1.1, origin="lower")
        ax2.axis("off")
        fig2.tight_layout(pad=0)
        st.pyplot(fig2)
        plt.close(fig2)

    with col3:
        st.caption("3. Hierarchical Classification Map")
        # RGB classification: Background dark, Penumbra Cyan, Umbra Magenta
        rgb_class = np.zeros((1024, 1024, 3), dtype=np.float32)
        rgb_class[disk_mask] = [0.08, 0.10, 0.15]  # Quiet Sun
        rgb_class[penumbra_mask > 0] = [0.1, 0.85, 0.7]  # Penumbra: Cyan
        rgb_class[umbra_mask > 0] = [1.0, 0.15, 0.35]   # Umbra: Bright Magenta/Red
        
        fig3, ax3 = plt.subplots(figsize=(4, 4), facecolor="#0e1117")
        ax3.imshow(rgb_class, origin="lower")
        ax3.axis("off")
        fig3.tight_layout(pad=0)
        st.pyplot(fig3)
        plt.close(fig3)

    with col4:
        st.caption("4. Annotated Detections & Bboxes")
        fig4, ax4 = plt.subplots(figsize=(4, 4), facecolor="#0e1117")
        ax4.imshow(norm_img, cmap="bone", vmin=0.4, vmax=1.1, origin="lower")
        if not features_df.empty:
            for _, row in features_df.iterrows():
                bx, by = row["bbox_x"], row["bbox_y"]
                bw, bh = row["bbox_w"], row["bbox_h"]
                rect = patches.Rectangle((bx, by), bw, bh, linewidth=1.0, edgecolor="#00ffcc", facecolor="none")
                ax4.add_patch(rect)
                ax4.plot(row["centroid_x"], row["centroid_y"], "r+", markersize=3)
        ax4.axis("off")
        fig4.tight_layout(pad=0)
        st.pyplot(fig4)
        plt.close(fig4)

    st.info(
        "💡 **Classification Legend:** Red/Magenta = **Umbral Core** ($I_{norm} < \\tau_u$, concentrated magnetic field), "
        "Cyan/Green = **Penumbral Halo** (intermediate filamentary zone), Dark Blue = **Quiet-Sun Photosphere**."
    )

# ---------------------------------------------------------
# Tab 2: Morphological Features
# ---------------------------------------------------------
with tab2:
    st.markdown("#### Extracted Region Descriptors per Connected Component")
    if features_df.empty:
        st.warning("No sunspots detected for this frame under current parameter thresholds.")
    else:
        st.dataframe(
            features_df[[
                "component_id", "centroid_x", "centroid_y", "area_pixels", "area_msh",
                "umbra_msh", "penumbra_msh", "up_ratio", "eccentricity", "solidity",
                "mean_intensity", "min_intensity"
            ]],
            use_container_width=True,
            height=300
        )

        c_f1, c_f2 = st.columns(2)
        with c_f1:
            fig_scat, ax_scat = plt.subplots(figsize=(6, 3.8))
            ax_scat.scatter(features_df["area_msh"], features_df["eccentricity"], color="#ffaa00", edgecolors="black", s=60, alpha=0.8)
            ax_scat.set_xlabel("Projected Area $A_f$ (ppm / µDisk)")
            ax_scat.set_ylabel("Eccentricity (from central moments)")
            ax_scat.set_title("Spot Area vs. Morphological Eccentricity")
            ax_scat.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig_scat)
            plt.close(fig_scat)

        with c_f2:
            fig_hist, ax_hist = plt.subplots(figsize=(6, 3.8))
            ax_hist.scatter(features_df["penumbra_msh"], features_df["umbra_msh"], color="#00ddff", edgecolors="black", s=60, alpha=0.8)
            ax_hist.set_xlabel("Penumbra Area (ppm / µDisk)")
            ax_hist.set_ylabel("Umbra Area (ppm / µDisk)")
            ax_hist.set_title("Umbra vs. Penumbra Physical Scaling")
            ax_hist.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig_hist)
            plt.close(fig_hist)

# ---------------------------------------------------------
# Tab 3: Temporal Tracking & SILSO
# ---------------------------------------------------------
with tab3:
    daily_df, silso_df, merged, metrics = load_temporal_results()
    if merged is not None and metrics is not None:

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Pearson Correlation (r)", f"{metrics['pearson_r']:.4f}")
        with col_m2:
            st.metric("Spearman Rank (rho)", f"{metrics['spearman_rho']:.4f}")
        with col_m3:
            st.metric("Goodness of Fit (R²)", f"{metrics['r_squared']:.4f}")
        with col_m4:
            st.metric("Statistical p-value", f"< 10⁻²¹")

        st.markdown("---")
        
        # Dual-Axis Time Series Plot
        fig_time, ax_t1 = plt.subplots(figsize=(12, 4.5))
        dates_merged = pd.to_datetime(merged["date"])
        
        color_af = "#ff8800"
        ax_t1.set_xlabel("Observation Date (Q4 2023)")
        ax_t1.set_ylabel("Projected Spot Area $A_f$ (ppm / µDisk)", color=color_af, fontweight="bold")
        l1 = ax_t1.plot(dates_merged, merged["total_area_msh"], color=color_af, lw=2.2, label="Pipeline Area $A_f$ (ppm / µDisk)")
        ax_t1.tick_params(axis="y", labelcolor=color_af)
        ax_t1.grid(True, linestyle="--", alpha=0.3)

        # Highlight current date
        curr_dt = pd.to_datetime(selected_date)
        ax_t1.axvline(curr_dt, color="#ff0055", linestyle=":", lw=2.5, label=f"Selected Day ({selected_date})")

        ax_t2 = ax_t1.twinx()
        color_sn = "#00bbff"
        ax_t2.set_ylabel("SILSO Sunspot Number S_N", color=color_sn, fontweight="bold")
        l2 = ax_t2.plot(dates_merged, merged["silso_sn"], color=color_sn, lw=2.0, linestyle="--", label="SILSO Sunspot Number S_N")
        ax_t2.tick_params(axis="y", labelcolor=color_sn)

        lines = l1 + l2 + [patches.Patch(color="#ff0055", label=f"Selected: {selected_date}")]
        labels = [l.get_label() for l in lines]
        ax_t1.legend(lines, labels, loc="upper left")
        ax_t1.set_title("91-Day Temporal Tracking: Pipeline Spot Area vs. Official SILSO Record", fontweight="bold")
        fig_time.tight_layout()
        st.pyplot(fig_time)
        plt.close(fig_time)

        # Scatter plot
        fig_scat2, ax_s2 = plt.subplots(figsize=(8, 4))
        ax_s2.scatter(merged["silso_sn"], merged["total_area_msh"], color="#22cc88", edgecolors="black", s=50, alpha=0.85)
        # Regression line
        slope, intercept = metrics["slope"], metrics["intercept"]
        x_vals = np.linspace(merged["silso_sn"].min(), merged["silso_sn"].max(), 100)
        ax_s2.plot(x_vals, slope * x_vals + intercept, "r-", lw=2, label=f"Fit: A_f = {slope:.2f}·S_N + {intercept:.1f}")
        ax_s2.set_xlabel("Official SILSO Sunspot Number S_N")
        ax_s2.set_ylabel("Projected Spot Area $A_f$ (ppm / µDisk)")
        ax_s2.set_title(f"91-Day Temporal Correlation & Regression Analysis (r = {metrics['pearson_r']:.3f}, R² = {metrics['r_squared']:.3f})")
        ax_s2.grid(True, linestyle="--", alpha=0.4)
        ax_s2.legend()
        fig_scat2.tight_layout()
        st.pyplot(fig_scat2)
        plt.close(fig_scat2)
    else:
        st.info("Run `python scripts/run_pipeline.py` to generate complete temporal tables.")

# ---------------------------------------------------------
# Tab 4: Benchmark Experiments & Syllabus
# ---------------------------------------------------------
with tab4:
    st.markdown("#### Scientific Benchmark Experiments")
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("##### Experiment A: Segmentation Comparison")
        if os.path.exists("results/tables/experiment_a_segmentation.csv"):
            df_a = pd.read_csv("results/tables/experiment_a_segmentation.csv")
            st.dataframe(df_a, use_container_width=True)
            st.caption("Evaluated across 5 photometric consensus reference masks. Top-Hat achieves highest Dice (0.830) and Recall (1.000).")
        
        st.markdown("##### Experiment B: Normalization Ablation")
        if os.path.exists("results/tables/experiment_b_ablation.csv"):
            df_b = pd.read_csv("results/tables/experiment_b_ablation.csv")
            st.dataframe(df_b, use_container_width=True)
            st.caption("Demonstrates catastrophic failure of thresholding without 5th-order polynomial limb-darkening flattening.")

    with col_e2:
        st.markdown("##### Syllabus Concept Mapping (IPCV Units I–V)")
        st.markdown("""
        | Syllabus Unit | Core Concept | Concrete Pipeline Implementation |
        | :--- | :--- | :--- |
        | **Unit I** | Image Sampling & Geometry | 32-bit FITS ingestion, WCS astrometry, spatial coordinate transformations |
        | **Unit II** | Enhancement & Morphology | Radial annular profiling, 5th-order flattening, White Top-Hat with disk SE, binary opening |
        | **Unit III** | Restoration & Segmentation | $3\\times3$ median denoising, Otsu variance minimization, adaptive Gaussian, $\\mu+3.5\\sigma$ threshold |
        | **Unit IV** | Contours & Component Analysis | Contour circle fitting for solar disk, 8-connectivity connected components, pore area filtering |
        | **Unit V** | Moments & Feature Descriptors | Central moments (eccentricity), convex hull solidity, hierarchical umbra/penumbra decomposition |
        """)
