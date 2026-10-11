"""
sunspot: Classical Computer Vision Pipeline for Automated Solar Activity Quantification.

Core modules:
- disk: Solar disk geometry detection and masking.
- normalize: Preprocessing, downsampling, and radial limb-darkening correction.
- segment: Multi-method classical segmentation (Otsu, Adaptive, Morphological Top-Hat).
- cleanup: Morphological post-processing (opening, hole filling, pore filtering).
- features: Connected components analysis and morphological feature extraction.
- metrics: Quantitative segmentation metrics (Dice, IoU, Precision, Recall).
- activity: Time series aggregation and external SILSO correlation analysis.
- pipeline: End-to-end single-image and batch orchestrator with caching.
"""

__version__ = "1.0.0"
