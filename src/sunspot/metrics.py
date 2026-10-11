"""
metrics.py
----------
Quantitative pixel-level segmentation evaluation metrics.

Implements standard academic computer-vision metrics:
1. Dice Similarity Coefficient (F1-score)
2. Intersection over Union (IoU / Jaccard Index)
3. Precision (Positive Predictive Value)
4. Recall (Sensitivity / True Positive Rate)
5. Overall Pixel Accuracy
"""

from typing import Dict
import numpy as np


def compute_segmentation_metrics(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    eps: float = 1e-7
) -> Dict[str, float]:
    """
    Compute binary segmentation evaluation metrics between prediction and ground truth.

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted binary mask (boolean or uint8 with values 0 or 1).
    true_mask : np.ndarray
        Ground truth binary mask (boolean or uint8 with values 0 or 1).
    eps : float, default 1e-7
        Small constant to avoid division by zero.

    Returns
    -------
    metrics : dict
        Dictionary containing 'dice', 'iou', 'precision', 'recall', 'pixel_accuracy',
        'tp', 'fp', 'fn', 'tn'.
    """
    p = (pred_mask > 0).astype(bool)
    t = (true_mask > 0).astype(bool)

    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum())
    tn = int(np.logical_and(~p, ~t).sum())

    total_pixels = tp + fp + fn + tn

    # Handle edge cases rigorously
    if (tp + fp == 0) and (tp + fn == 0):
        # Both masks are completely empty -> perfect agreement
        dice = 1.0
        iou = 1.0
        precision = 1.0
        recall = 1.0
    else:
        dice = float(2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    pixel_accuracy = float(tp + tn) / float(total_pixels) if total_pixels > 0 else 1.0

    return {
        "dice": round(dice, 4),
        "iou": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "pixel_accuracy": round(pixel_accuracy, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
