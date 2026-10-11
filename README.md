# Automated Solar Activity Quantification Using Classical Computer Vision

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 19 Passed](https://img.shields.io/badge/tests-19%20passed-brightgreen.svg)]()
[![Paradigm: Classical CV](https://img.shields.io/badge/paradigm-100%25%20classical%20CV-orange.svg)]()
[![Dataset: SDO/HMI Q4 2023](https://img.shields.io/badge/dataset-SDO%2FHMI%20(91%20days)-blue.svg)]()

> **An Image Processing and Computer Vision (IPCV) Course Project**  
> A fully autonomous, interpretable, and computationally lightweight classical computer vision pipeline in Python that ingests full-disk solar continuum observations from NASA's Solar Dynamics Observatory (SDO), segments sunspots using mathematical morphology, extracts quantitative geometric features, and derives a solar activity index rigorously validated against the international SILSO benchmark.

---

## Table of Contents

1. [Executive Summary & Motivation](#1-executive-summary--motivation)
2. [The Solar Physics Context: Why Tracking the Sun Matters](#2-the-solar-physics-context-why-tracking-the-sun-matters)
3. [Computer Vision Challenges & Our Technical Innovations](#3-computer-vision-challenges--our-technical-innovations)
4. [Scientific Formulation: Fractional Area ($A_f$) vs. Wolf Number ($R_z$)](#4-scientific-formulation-fractional-area-a_f-vs-wolf-number-r_z)
5. [Classical Vision Pipeline Architecture](#5-classical-vision-pipeline-architecture)
6. [IPCV Syllabus Alignment Map (Units I–V)](#6-ipcv-syllabus-alignment-map-units-iv)
7. [Experimental Results & Benchmarks](#7-experimental-results--benchmarks)
   - [Experiment A: 3-Way Classical Segmentation Benchmark](#experiment-a-3-way-classical-segmentation-benchmark)
   - [Experiment B: Solar Limb-Darkening Ablation Study](#experiment-b-solar-limb-darkening-ablation-study)
   - [Experiment C: Hyperparameter Sensitivity Analysis](#experiment-c-hyperparameter-sensitivity-analysis)
   - [Experiment D: 91-Day Solar Cycle 25 Temporal Validation against SILSO](#experiment-d-91-day-solar-cycle-25-temporal-validation-against-silso)
8. [Repository Structure](#8-repository-structure)
9. [Installation & Reproduction](#9-installation--reproduction)
10. [Scientific References](#10-scientific-references)

---

## 1. Executive Summary & Motivation

Sunspots are intensely magnetized, relatively cool structures on the solar photosphere. For over four centuries, tracking sunspot count and surface area has been humanity’s primary method for measuring solar variability, predicting solar flares, and understanding the 11-year solar activity cycle.

In modern space science, the **Solar Dynamics Observatory (SDO)** continuously captures full-disk 4096×4096 continuum images of the photosphere every 45 seconds. While deep learning models (e.g., U-Net, Mask R-CNN) are popular today, they suffer from three fundamental drawbacks for this application:
1. **Black-box opacity:** Neural network activations lack physical interpretability and struggle with out-of-distribution observational artifacts.
2. **Computational overhead:** Full 4K neural inference requires high-end GPUs, preventing real-time edge processing on modest hardware.
3. **Training data bottlenecks:** Pixel-level ground-truth sunspot annotations are scarce and subject to high inter-annotator variance.

**What is Special About Our Approach:**
* **100% Classical Computer Vision:** We rely strictly on deterministic spatial filtering, mathematical morphology, intensity normalization, and connected-component analysis.
* **Zero Deep Learning Dependencies:** No PyTorch, no TensorFlow, no GPU required.
* **Blazing Fast Laptop Compute:** Optimized using spatial downsampling and vectorized OpenCV morphological primitives, processing an entire 4K full-disk observation in **$< 80\text{ ms}$** on a standard CPU.
* **Physical Grounding:** Rather than attempting an arbitrary heuristic reconstruction of the observer-dependent Wolf Number, we extract **Total Fractional Sunspot Area ($A_f$, in Millionths of the Solar Hemisphere)**—a direct physical observable that correlates with the official international record at $r = 0.814$ ($p < 10^{-10}$).
* **100% Offline Capable:** The complete pipeline, tests, and demonstration notebook execute locally with zero external API calls or network access.

---

## 2. The Solar Physics Context: Why Tracking the Sun Matters

### 2.1 The Solar Engine & Active Regions
The Sun is a dynamic ball of plasma governed by magnetohydrodynamics. Deep within the convective zone, differential rotation stretches magnetic field lines into powerful flux tubes. When these magnetic ropes buoyantly erupt through the photosphere, they inhibit convective heat transport from the solar interior. 
* The surrounding undisturbed photosphere radiates at approximately **5,778 K** ($\approx 40,000\text{ DN}$ in SDO continuum counts).
* The inhibited active regions radiate at only **3,500–4,500 K**, appearing visually as dark features termed **sunspots**.
* A mature sunspot exhibits two distinct anatomical zones:
  - **Umbra:** The intensely dark inner core where vertical magnetic flux is strongest ($B \approx 2,000\text{--}3,500\text{ G}$).
  - **Penumbra:** The filamentary outer halo where magnetic field lines bend outward horizontally into the canopy.

### 2.2 Space Weather & Societal Impact
Sunspot complexes are the physical cradles of **solar flares** and **coronal mass ejections (CMEs)**. When twisted magnetic field lines reconnect violently above sunspot groups, they blast billions of tons of magnetized plasma across the solar system. Severe space weather events can:
- Induce geomagnetically induced currents that collapse terrestrial power grids.
- Ionize Earth's upper atmosphere, disrupting HF military radio and satellite GPS navigation.
- Expose satellites and astronauts to lethal radiation doses.

### 2.3 Solar Cycle 25 Maximum
Solar activity fluctuates over a semi-regular 11-year cycle. Our selected observing window (**October 1 – December 31, 2023**) captures 91 consecutive days during the rapid ascent phase toward the **Solar Cycle 25 maximum**, characterized by large multipolar active regions (e.g., active region complexes AR 13451, AR 13500) and dramatic day-to-day variations in sunspot area.

---

## 3. Computer Vision Challenges & Our Technical Innovations

Applying classical computer vision to astronomical full-disk solar imagery presents several distinct challenges that break textbook segmentation algorithms:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             COMPUTER VISION CHALLENGES                           │
├─────────────────────────┬───────────────────────────────┬────────────────────────┤
│ Challenge               │ Why Naive Methods Fail        │ Our Classical Solution │
├─────────────────────────┼───────────────────────────────┼────────────────────────┤
│ 1. Solar Limb           │ Photosphere intensity drops   │ 5th-order radial       │
│    Darkening            │ 40% from center to edge.      │ polynomial profile     │
│                         │ Global thresholding labels    │ fit I(ρ) + division to │
│                         │ the entire limb as a sunspot! │ flatten quiet Sun ~1.0 │
├─────────────────────────┼───────────────────────────────┼────────────────────────┤
│ 2. Extreme Class        │ Sunspots occupy < 0.5% of the │ Morphological White    │
│    Imbalance            │ disk; background is 99.5%.    │ Top-Hat transform with │
│                         │ Otsu thresholding collapses.  │ statistical cutoff     │
│                         │ Standard accuracy is useless. │ (μ + 3.5σ) + Dice/IoU  │
├─────────────────────────┼───────────────────────────────┼────────────────────────┤
│ 3. Granulation Noise    │ Photospheric convection       │ Median filtering +     │
│    vs. Faint Pores      │ creates ±5-8% intensity       │ physical minimum area  │
│                         │ ripple across the entire Sun. │ pore rejection         │
│                         │ Gradient edge filters fire.   │ (Amin ≥ 12 MSH)        │
├─────────────────────────┼───────────────────────────────┼────────────────────────┤
│ 4. Limb Foreshortening  │ Perspective squashes spots    │ Strict analysis mask   │
│    & Edge Divergence    │ near the limb; optical depth  │ buffer (r ≤ 0.92 R☉)   │
│                         │ gradients diverge rapidly.    │ isolates valid disk.   │
└─────────────────────────┴───────────────────────────────┴────────────────────────┘
```

### Detailed Breakdown of Solutions

#### Challenge 4: Limb Foreshortening & Geometric Masking
* **Mask Geometry (`disk.py`):** We employ an analytical circular disk mask with radial cutoff $r_{fraction} = 0.92$.
* **Area Retention:** The retained disk covers $(0.92)^2 \approx 84.64\%$ of the total projected circular disk area, excluding the outermost $15.36\%$ annular perimeter ($\mu = \cos \theta < 0.392$).
* **Scientific Assessment & Trade-offs:**
  - *Mitigation:* Near the extreme solar limb, the optical depth gradient steepens sharply ($\frac{dI}{dr} \to -\infty$), causing polynomial background models to diverge and producing spurious false-positive rings. The $0.92 R_\odot$ buffer cleanly eliminates this edge degradation.
  - *Trade-off:* Spots located within $\approx 1\text{--}2$ days of limb entry (east) or exit (west) lie in the excluded $15.36\%$ annular margin and are temporarily omitted from image-derived area until rotating past $\rho \le 0.92$. Because geometric foreshortening in that extreme zone ($\mu < 0.4$) compresses spot profiles to near-zero radial width, planar 2D segmentation in that margin is inherently unreliable without magnetographic field line modeling.

---

## 4. Scientific Formulation: Projected Disk Area Fraction ($A_f$) vs. Wolf Number ($R_z$)

### Why We Reject Direct Wolf Number ($R_z$) Reconstruction
The historical International Sunspot Number (maintained by SILSO since 1818) is defined as:
$$R_z = k(10g + s)$$
where $g$ is the number of active spot groups, $s$ is the total number of individual spots, and $k$ is an observer-specific calibration factor.

Attempting to compute $R_z$ using automated classical computer vision is scientifically problematic:
1. **Component Over-Segmentation:** A single large active region with 4 umbral cores inside a common penumbra will be counted as either 1 spot or 5 spots depending on minor threshold fluctuations.
2. **Magnetic Bipolar Grouping Ambiguity:** Grouping spots into active regions ($g$) requires magnetic polarity orientation (Hale's polarity law from line-of-sight magnetograms). Spatial clustering heuristics (e.g., Euclidean distance or DBSCAN) frequently merge distinct neighboring active regions or fracture complex multipolar centers.
3. **Circular Calibration:** Tuning $k$ via regression against SILSO introduces circular reasoning.

### The Objective Observable: Projected Solar Disk Area Fraction ($A_f$)
Instead, we extract **Projected Solar Disk Area Fraction ($A_f$)**:
$$A_f = \frac{\sum_{i} \text{Area}_{i}}{\pi R_\odot^2} \times 10^6 \quad [\text{parts-per-million of visible disk / } \mu\text{Disk}]$$

* **Physical Formulation & Important Distinction:**
  - $A_f$ is an objective, deterministic planar measurement of segmented sunspot pixels relative to the projected solar disk.
  - **Distinction from True Spherical MSH:** In classical solar catalogs (e.g. Royal Greenwich Observatory, NOAA/SOON), Millionths of the Solar Hemisphere (MSH) is defined over the 3D spherical surface of the visible hemisphere ($2\pi R_\odot^2$) and requires $1/\cos \theta$ foreshortening de-projection. Because our 2D classical pipeline does not apply spherical de-projection, our metric $A_f$ strictly measures **projected disk area fraction in parts-per-million ($\mu\text{Disk}$ / ppm)**. We report this transparently rather than applying empirical de-projection multipliers that diverge toward the solar limb.
  - In solar physics literature (Watson & Fletcher 2011, Carvalho et al. 2020), $A_f$ is documented to correlate with $R_z$ at $r \approx 0.80\text{--}0.92$ over solar cycle timescales.

---

## 5. Classical Vision Pipeline Architecture

The pipeline processes raw SDO/HMI continuum observations through seven decoupled, typed modules:

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Raw SDO/HMI Continuum FITS Ingestion (4096 x 4096 Float Array)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ 2. Geometry & Astrometric Masking (src/sunspot/disk.py)                │
│    • Extract CRPIX1, CRPIX2, RSUN_OBS, CDELT1 from FITS WCS header      │
│    • Classical contour enclosing circle fallback if WCS missing        │
│    • Generate circular disk mask with boundary buffer (r ≤ 0.92 R☉)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ 3. Downsampling & Limb Normalization (src/sunspot/normalize.py)        │
│    • cv2.INTER_AREA downscale to 1024 x 1024 (sub-100ms execution)     │
│    • 3x3 median filter for cosmic ray & high-frequency suppression     │
│    • 150-bin annular median profile + 5th-order polynomial fit         │
│    • Flatten quiet-Sun photosphere intensity to ~1.000 ± 0.017         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ 4. Morphological Segmentation (src/sunspot/segment.py)                 │
│    • Image inversion: I_inv = max(0, 1.05 - I_norm)                    │
│    • cv2.MORPH_TOPHAT with circular structuring element (r = 21 px)    │
│    • Statistical standard deviation cutoff: τ = μ + 3.5σ               │
│    • Optional per-region adaptive umbra segmentation: τ_u = μ_c - σ_c  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ 5. Morphological Post-Processing (src/sunspot/cleanup.py)              │
│    • Binary opening (3x3 ellipse) to sever thin noise bridges          │
│    • Binary hole filling to consolidate fractured umbra/penumbra       │
│    • Minimum area pore rejection: discard components < 8 px (< 12 MSH) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ 6. Connected Components & Feature Extraction (src/sunspot/features.py) │
│    • 8-connectivity labeling (cv2.connectedComponentsWithStats)        │
│    • Morphological table: Centroid, Area (px & MSH), Bbox,             │
│      Eccentricity (from central moments μ20, μ02, μ11), Solidity       │
│    • Disk aggregate: Spot Count, Total Fractional Area (Af)            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ 7. Time Series & External Validation (src/sunspot/activity.py)         │
│    • Ingest official SILSO daily record (SN_d_tot_V2.0.csv)            │
│    • Date-alignment across all 91 observation days (Q4 2023)           │
│    • Pearson r, Spearman ρ, OLS regression slope & R² analysis         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 6. IPCV Syllabus Alignment Map (Units I–V)

The project directly illustrates core concepts across all five units of the IPCV course curriculum:

| Syllabus Unit | Core Theoretical Concept | Project Implementation & Justification | Source Module |
| :--- | :--- | :--- | :--- |
| **Unit I: Digital Image Fundamentals** | Sampling, Quantization, Pixel Geometry & Coordinate Systems | Ingestion of 16-bit floating-point astronomical FITS data; sub-pixel WCS astrometric center and radius mapping; Nyquist-compliant downsampling ($4096 \to 1024$) with area anti-aliasing. | `disk.py`<br>`normalize.py` |
| **Unit II: Histogram & Spatial Enhancement** | Intensity Histograms, Non-Linear Spatial Filtering | Annular median intensity profile construction; quiet-Sun vs. sunspot histogram analysis; 3×3 median filtering for cosmic ray noise suppression without edge blurring. | `normalize.py` |
| **Unit II: Mathematical Morphology** | Structuring Elements, Dilation, Erosion, Opening, Top-Hat Transform | Circular structuring element synthesis ($B_{disk}, r=21\text{ px}$); morphological opening to sever noise spurs; binary hole filling; Morphological White Top-Hat residual extraction. | `segment.py`<br>`cleanup.py` |
| **Unit III: Thresholding & Segmentation** | Global vs. Local Adaptive vs. Statistical Thresholding | Comparative evaluation of Global Otsu thresholding, Local Gaussian Adaptive thresholding, and Statistical Standard Deviation thresholding ($\mu + k\sigma$). | `segment.py` |
| **Unit IV: Edge Detection & Geometric Contours** | Gradient Operators, Contour Analysis, Circle Fitting | Educational fallback disk detector utilizing thresholding, morphological closing, and contour minimum enclosing circle fitting. | `disk.py` |
| **Unit V: Region Properties & Feature Extraction** | Connected Components, Spatial Moments, Shape Descriptors | 8-connected components labeling; equivalent radius; eccentricity derived from eigenvalues of second-order central spatial moments ($\mu_{20}, \mu_{02}, \mu_{11}$); solidity via convex hull contour ratio. | `features.py` |

---

## 7. Experimental Results & Benchmarks

All experiments were executed on the full 91-day SDO/HMI continuum observation dataset (October 1 – December 31, 2023).

### Experiment A: 3-Way Classical Segmentation Benchmark
We evaluated the three classical segmentation paradigms against photometric consensus reference masks across 5 representative dates spanning diverse solar morphology (high activity, distributed regions, isolated spots, quiet photosphere, and limb transits).

> [!IMPORTANT]
> **Reference Mask Methodology Note:** The reference masks used here are *photometric consensus reference masks* generated algorithmically via conservative morphological top-hat filtering combined with multi-stage spatial cleanup. They serve as a standardized, reproducible benchmark for comparing classical algorithmic variants (e.g. Otsu vs. Adaptive vs. Top-Hat) under identical assumptions, but are **not** independent human ground truth or external observer annotations. Independent segmentation accuracy cannot be claimed without external manual annotations.

**Quantitative Results (`results/tables/experiment_a_segmentation.csv`):**

| Segmentation Method | Mean Dice | Mean IoU | Mean Precision | Mean Recall | Failure Mode / Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Global Otsu (Baseline 1)** | 0.1757 | 0.1478 | 0.2043 | 0.9435 | **Catastrophic Failure:** Unimodal solar histogram causes Otsu to classify granulation ripples across the entire disk (>200,000 px). |
| **Local Adaptive (Baseline 2)** | 0.6258 | 0.5172 | 0.8000* | 0.5172 | **Penumbra Truncation:** Zero false positives on active days, but local window truncates diffuse outer penumbral halos, reducing recall to 51.7%. |
| **Morphological Top-Hat (Ours)** | **0.8296** | **0.7365** | **0.7365** | **1.0000** | **Optimal Performance:** Perfect 100% recall capturing subtle penumbral boundaries and high Dice overlap (up to **0.877** on active days). |

*\*Note on Precision:* Local Adaptive precision is 0.8000 across the 5 benchmark dates (1.0000 on the 4 active days and 0.0000 on the quiet-Sun test case where $TP=0, FP=0$ after correcting a zero-division edge case that formerly reported 1.0000).

![Segmentation Comparison](results/figures/fig2_segmentation_comparison.png)
*Figure 2: Visual comparison of Consensus Reference (lime), Morphological Top-Hat (cyan), Global Otsu (red), and Local Adaptive (orange) on active region complex AR 13451.*

---

### Experiment B: Solar Limb-Darkening Ablation Study
We quantified the scientific impact of radial polynomial limb-darkening correction by comparing Morphological Top-Hat segmentation under two conditions:
1. **Normalized (Limb Corrected):** 5th-order radial polynomial division.
2. **Ablated (Raw Uncorrected):** Scaled raw continuum intensity without radial correction.

**Quantitative Results (`results/tables/experiment_b_ablation.csv`):**

| Preprocessing Condition | Mean Dice | Mean IoU | Precision | Mean Limb False Positives ($\rho > 0.85$) |
| :--- | :---: | :---: | :---: | :---: |
| **Ablated (Raw Uncorrected)** | 0.6703 | 0.5612 | 0.6577 | 250.4 px (Spurious edge ring) |
| **Normalized (Limb Corrected)** | **0.8296** | **0.7365** | **0.7365** | **74.6 px** (70.2% false positive reduction) |

![Ablation Study](results/figures/fig3_normalization_ablation.png)
*Figure 3: Normalization ablation. Without limb-darkening correction, radial intensity drop creates an artificial ring of false detections near the edge. Our polynomial normalization eliminates this artifact.*

---

### Experiment C: Hyperparameter Sensitivity Analysis
We evaluated the stability of the morphological pipeline across structuring element radii ($r \in [11, 15, 21, 27, 35]\text{ px}$) and statistical cutoff multipliers ($k_\sigma \in [2.5, 3.0, 3.5, 4.0]$).

**Key Findings (`results/tables/experiment_c_sensitivity.csv`):**
* Structuring element radius $r \ge 21\text{ px}$ achieves asymptotic stability: sunspots are smaller than $r=21\text{ px}$ at $1024\times1024$, ensuring background estimation does not erode spot interiors.
* $k_\sigma = 3.5$ provides the optimal physical boundary cutoff: it rejects convective granulation ripples ($\pm 5\text{--}8\%$) while preserving outer penumbrae ($\approx 12\text{--}15\%$ intensity drop).

![Parameter Sensitivity](results/figures/fig4_parameter_sensitivity.png)
*Figure 4: Sensitivity of sunspot count and total fractional area ($A_f$) to structuring element radius and threshold multiplier $k_\sigma$.*

---

### Experiment D: 91-Day Solar Cycle 25 Temporal Validation against SILSO
We processed all **91 daily observations** spanning October 1 to December 31, 2023, date-aligned the extracted Total Projected Fractional Sunspot Area ($A_f$, in $\mu\text{Disk}$ / ppm) with the official Royal Observatory of Belgium International Sunspot Number ($S_N$), and performed statistical correlation analysis.

**Statistical Summary (`results/tables/merged_activity_silso.csv`):**
* **Total Observation Days:** $N = 91$ days (98.9% continuous daily coverage)
* **Pearson Correlation Coefficient ($r$):** **0.8139** ($p = 1.05 \times 10^{-22}$)
* **Spearman Rank Correlation ($\rho$):** **0.8156** ($p = 7.28 \times 10^{-23}$)
* **Coefficient of Determination ($R^2$):** **0.6625**
* **Linear Regression Model:**
  $$S_N = 0.041 \cdot A_f + 53.4$$

**Supplementary Association Diagnostic: First-Difference Analysis:**
To test whether the high correlation is purely an artifact of shared low-frequency solar rotation and cycle trends, we computed the day-to-day first differences ($\Delta A_f(t) = A_f(t) - A_f(t-1)$ and $\Delta S_N(t) = S_N(t) - S_N(t-1)$):
* **First-Difference Pearson Correlation ($r_\Delta$):** **0.4495** ($p = 8.80 \times 10^{-6}$)
* **First-Difference Spearman Rank Correlation ($\rho_\Delta$):** **0.4259** ($p = 2.85 \times 10^{-5}$)
* **Scientific Interpretation:** The statistically significant positive first-difference correlation ($p < 10^{-5}$) confirms that day-to-day area rate changes track international activity variations beyond pure seasonal drift. However, because $r_\Delta \approx 0.450$ is substantially lower than $r = 0.814$, this metric is presented as a *supplementary association diagnostic*, not proof of validated day-to-day causal tracking.

![SILSO Temporal Validation](results/figures/fig5_silso_temporal_validation.png)
*Figure 5: 91-Day temporal validation. Top panel: Dual-axis time series of classical image area $A_f$ [ppm / \mu Disk] (blue) and official SILSO Sunspot Number $S_N$ (red). Bottom panel: Linear regression scatter plot ($R^2 = 0.663, r = 0.814$).*

![Preprocessing Pipeline](results/figures/fig1_preprocessing_pipeline.png)
*Figure 1: Full preprocessing sequence: (a) Raw 4K SDO continuum with limb darkening, (b) Astrometric disk mask, (c) 150-bin annular median profile and 5th-order polynomial fit, (d) Flattened quiet-Sun photosphere.*

---

## 8. Repository Structure

```
IPCV_CBP/
├── data/
│   ├── raw_fits/                # Multi-cadence SDO/HMI continuum FITS files
│   ├── daily_fits/              # 91 daily FITS files (~12:00:00 TAI) [1.4 GB]
│   ├── groundtruth/             # 5 photometric consensus reference masks (.npz & .png)
│   ├── SN_d_tot_V2.0.csv        # Official daily SILSO international sunspot records
│   ├── download_data.py         # SunPy network acquisition utility
│   ├── download_jsoc.py         # Direct JSOC continuum downloader
│   └── select_daily.py          # Daily cadence selector (12:00:00 TAI)
├── src/
│   └── sunspot/
│       ├── __init__.py          # Package initialization
│       ├── disk.py              # WCS astrometric parsing & circular disk masking
│       ├── normalize.py         # Downsampling & 5th-order radial polynomial flattening
│       ├── segment.py           # 3-way classical segmentation suite (Top-Hat, Otsu, Adaptive)
│       ├── cleanup.py           # Opening, hole filling, and minimum-area pore filtering
│       ├── features.py          # Connected components & geometric properties table
│       ├── metrics.py           # Pixel-level segmentation metrics (Dice, IoU, Precision, Recall)
│       ├── activity.py          # SILSO parser, correlation analysis, and publication plotting
│       └── pipeline.py          # End-to-end batch orchestrator with NPZ mask caching
├── scripts/
│   ├── create_reference_masks.py# Generates photometric consensus reference masks
│   ├── run_pipeline.py          # Full 91-day batch pipeline runner (< 15 seconds cached)
│   └── run_experiments.py       # Executes Experiments A, B, C, D and exports figures
├── notebooks/
│   └── demo.ipynb               # Self-contained, 100% offline interactive presentation
├── app.py                       # Interactive Streamlit Web Dashboard (live tuning & overlays)
├── results/
│   ├── figures/                 # Generated publication-quality figures (Figs 1–5)
│   ├── masks/                   # Cached segmentation masks (*.npz)
│   └── tables/                  # Experimental benchmark CSV tables
├── tests/
│   ├── test_disk.py             # Unit tests for disk geometry and masking
│   ├── test_normalize.py        # Unit tests for limb-darkening profile fitting
│   ├── test_segment.py          # Unit tests for morphological segmentation & umbra decomposition
│   ├── test_features.py         # Unit tests for connected component & umbra features
│   ├── test_metrics.py          # Unit tests for Dice/IoU calculation
│   ├── test_activity.py         # Unit tests for SILSO correlation math
│   └── test_pipeline.py         # Integration test for SunspotPipeline orchestrator
├── PHASE_0_DISCOVERY_REPORT.md  # Detailed initial feasibility and reference audit report
├── PROJECT_EXECUTION_PLAN.md    # Complete end-to-end roadmap and signed-off specifications
├── pytest.ini                   # Pytest test discovery configuration
├── requirements.txt             # Python package dependencies
└── README.md                    # Project documentation (this file)
```

---

## 9. Installation & Reproduction

### Prerequisites
* Python 3.10+
* Windows, macOS, or Linux (runs 100% locally on standard CPU)

### Setup
```bash
# 1. Clone repository and navigate to root
git clone <repo-url>
cd IPCV_CBP

# 2. Install dependencies
pip install -r requirements.txt
```

### Reproduce Everything in Under 2 Minutes

```bash
# 1. Run the test suite (19 unit & integration tests)
pytest -v tests/

# 2. Run the 91-day automated pipeline
python scripts/run_pipeline.py

# 3. Re-run all 4 scientific experiments and re-generate all figures
python scripts/run_experiments.py
```

### Interactive Live Web Dashboard & Notebook
Launch the full interactive Streamlit web dashboard:
```bash
streamlit run app.py
```
Or launch Jupyter Notebook to step through the offline presentation:
```bash
jupyter notebook notebooks/demo.ipynb
```

---

## 10. Scientific References

1. **Curto, J. J., Blanca, M., & Martínez, E. (2008).** *Automatic Sunspots Detection on Full-Disk Solar Images using Mathematical Morphology.* Solar Physics, 250(2), 411–429. [DOI: 10.1007/s11207-008-9224-6](https://doi.org/10.1007/s11207-008-9224-6)
2. **Zharkov, S., Zharkova, V. V., Ipson, S. S., & Benkhalil, A. (2005).** *An automated technique for sunspot detection and extraction with SOHO/MDI.* Solar Physics, 228(1), 377–389. [DOI: 10.1007/s11207-005-5005-7](https://doi.org/10.1007/s11207-005-5005-7)
3. **Watson, F. T., & Fletcher, L. (2011).** *Automated sunspot detection and the evolution of sunspot magnetic fields during solar cycle 23.* Proceedings of the International Astronomical Union, 7(S273), 51–55. [DOI: 10.1017/S1743921311014992](https://doi.org/10.1017/S1743921311014992)
4. **Carvalho, S., Gomes, S., Barata, T., et al. (2020).** *Comparison of automatic methods to detect sunspots in the Coimbra Observatory spectroheliograms.* Astronomy and Computing, 32, 100385. [DOI: 10.1016/j.ascom.2020.100385](https://doi.org/10.1016/j.ascom.2020.100385)
5. **Zhao, C., Lin, G. H., Deng, Y. Y., et al. (2016).** *Automatic Recognition of Sunspots in HSOS Full-Disk Solar Images.* Publications of the Astronomical Society of Australia, 33, e018. [DOI: 10.1017/pasa.2016.17](https://doi.org/10.1017/pasa.2016.17)
6. **SILSO World Data Center (2023).** *The International Sunspot Number.* Royal Observatory of Belgium, Brussels. `SN_d_tot_V2.0.csv`.
