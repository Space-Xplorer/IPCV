"""
test_metrics.py
---------------
Unit tests for pixel-level segmentation evaluation metrics.
"""

import numpy as np
import pytest
from sunspot.metrics import compute_segmentation_metrics


def test_perfect_overlap():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:30, 10:30] = 1

    metrics = compute_segmentation_metrics(mask, mask)
    assert metrics["dice"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_zero_overlap():
    pred = np.zeros((50, 50), dtype=np.uint8)
    true = np.zeros((50, 50), dtype=np.uint8)
    pred[0:10, 0:10] = 1
    true[40:50, 40:50] = 1

    metrics = compute_segmentation_metrics(pred, true)
    assert metrics["dice"] == 0.0
    assert metrics["iou"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_partial_overlap():
    pred = np.zeros((10, 10), dtype=np.uint8)
    true = np.zeros((10, 10), dtype=np.uint8)
    # pred has 4 pixels, true has 4 pixels, 2 overlap
    pred[0:2, 0:2] = 1
    true[0:2, 1:3] = 1

    # Overlap (TP) = 2, FP = 2, FN = 2
    # Dice = 2*2 / (2*2 + 2 + 2) = 4 / 8 = 0.5
    # IoU = 2 / (2 + 2 + 2) = 2 / 6 = 0.3333
    metrics = compute_segmentation_metrics(pred, true)
    assert np.isclose(metrics["dice"], 0.5, atol=1e-3)
    assert np.isclose(metrics["iou"], 0.3333, atol=1e-3)
    assert np.isclose(metrics["precision"], 0.5, atol=1e-3)
    assert np.isclose(metrics["recall"], 0.5, atol=1e-3)


def test_both_empty_masks():
    empty_pred = np.zeros((20, 20), dtype=np.uint8)
    empty_true = np.zeros((20, 20), dtype=np.uint8)
    metrics = compute_segmentation_metrics(empty_pred, empty_true)
    assert metrics["dice"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_empty_prediction_nonempty_groundtruth():
    empty_pred = np.zeros((20, 20), dtype=np.uint8)
    true_mask = np.zeros((20, 20), dtype=np.uint8)
    true_mask[5:10, 5:10] = 1
    metrics = compute_segmentation_metrics(empty_pred, true_mask)
    assert metrics["dice"] == 0.0
    assert metrics["iou"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_nonempty_prediction_empty_groundtruth():
    pred_mask = np.zeros((20, 20), dtype=np.uint8)
    pred_mask[5:10, 5:10] = 1
    empty_true = np.zeros((20, 20), dtype=np.uint8)
    metrics = compute_segmentation_metrics(pred_mask, empty_true)
    assert metrics["dice"] == 0.0
    assert metrics["iou"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0

