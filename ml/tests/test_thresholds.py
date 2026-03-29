from __future__ import annotations

import numpy as np

from ml.src.thresholds import (
    best_f1_threshold,
    fixed_threshold,
    percentile_threshold,
    threshold_sweep_dataframe,
)


def test_fixed_threshold_returns_value():
    assert fixed_threshold(0.01) == 0.01


def test_percentile_threshold_returns_expected_value():
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    thr = percentile_threshold(scores, percentile=50)
    assert abs(thr - 0.25) < 1e-9


def test_best_f1_threshold_finds_good_separator():
    y_true = np.array([0, 0, 1, 1], dtype=int)
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)

    res = best_f1_threshold(y_true, scores, n_steps=100)

    assert 0.2 <= res.threshold <= 0.8
    assert abs(res.f1 - 1.0) < 1e-9


def test_threshold_sweep_dataframe_has_expected_columns():
    y_true = np.array([0, 0, 1, 1], dtype=int)
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)

    df = threshold_sweep_dataframe(y_true, scores, n_steps=20)

    assert len(df) > 0
    assert list(df.columns) == ["threshold", "precision", "recall", "f1"]
