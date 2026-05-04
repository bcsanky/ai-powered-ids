from __future__ import annotations

import numpy as np
import pytest

from ml.src.eval import compute_classification_metrics
from ml.src.train_ae import compute_metrics


def make_metric_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.array([0] * 10 + [1] * 10)
    y_pred = np.array([0] * 8 + [1] * 2 + [0] * 3 + [1] * 7)
    scores = np.linspace(0.0, 1.0, num=len(y_true))
    return y_true, y_pred, scores


@pytest.mark.parametrize(
    "metric_fn",
    [
        compute_metrics,
        compute_classification_metrics,
    ],
)
def test_extended_classification_metrics(metric_fn):
    y_true, y_pred, scores = make_metric_inputs()

    metrics = metric_fn(y_true, y_pred, scores)

    assert metrics["tn"] == 8
    assert metrics["fp"] == 2
    assert metrics["fn"] == 3
    assert metrics["tp"] == 7
    assert metrics["false_positive_rate"] == pytest.approx(2 / 10)
    assert metrics["false_negative_rate"] == pytest.approx(3 / 10)
    assert metrics["true_positive_rate"] == pytest.approx(7 / 10)
    assert metrics["true_negative_rate"] == pytest.approx(8 / 10)
    assert metrics["alert_count"] == 9
