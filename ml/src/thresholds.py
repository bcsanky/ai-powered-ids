from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


@dataclass
class ThresholdResult:
    name: str
    threshold: float
    precision: float
    recall: float
    f1: float


def fixed_threshold(value: float) -> float:
    if value < 0:
        raise ValueError("A fixed threshold nem lehet negatív.")
    return float(value)


def percentile_threshold(scores: np.ndarray, percentile: float = 95.0) -> float:
    if scores.size == 0:
        raise ValueError("Üres scores tömbből nem lehet percentilis küszöböt számolni.")
    if not 0 <= percentile <= 100:
        raise ValueError("A percentilisnek 0 és 100 közé kell esnie.")

    scores = np.asarray(scores, dtype=np.float64)
    return float(np.percentile(scores, percentile))


def evaluate_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    name: str,
) -> ThresholdResult:
    y_pred = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return ThresholdResult(
        name=name,
        threshold=float(threshold),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


def best_f1_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_steps: int = 200,
    name: str = "f1_optimum",
) -> ThresholdResult:
    if scores.size == 0:
        raise ValueError("Üres scores tömbből nem lehet F1-optimum küszöböt keresni.")
    if y_true.size == 0:
        raise ValueError("Üres y_true tömbből nem lehet F1-optimum küszöböt keresni.")
    if scores.shape[0] != y_true.shape[0]:
        raise ValueError("A scores és y_true hossza nem egyezik.")
    if n_steps <= 1:
        raise ValueError("Az n_steps értékének 1-nél nagyobbnak kell lennie.")

    s_min = float(np.min(scores))
    s_max = float(np.max(scores))

    if s_min == s_max:
        return evaluate_threshold(y_true, scores, s_min, name=name)

    candidates = np.linspace(s_min, s_max, n_steps)

    best = None
    for thr in candidates:
        current = evaluate_threshold(y_true, scores, float(thr), name=name)

        if best is None:
            best = current
            continue

        if current.f1 > best.f1:
            best = current
            continue

        if current.f1 == best.f1 and current.precision > best.precision:
            best = current
            continue

        if (
            current.f1 == best.f1
            and current.precision == best.precision
            and current.threshold > best.threshold
        ):
            best = current

    return best


def threshold_sweep_dataframe(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_steps: int = 200,
) -> pd.DataFrame:
    if scores.size == 0:
        raise ValueError("Üres scores tömbből nem lehet threshold sweep-et készíteni.")

    s_min = float(np.min(scores))
    s_max = float(np.max(scores))

    if s_min == s_max:
        rows = [evaluate_threshold(y_true, scores, s_min, name="sweep")]
        return pd.DataFrame(
            [
                {
                    "threshold": rows[0].threshold,
                    "precision": rows[0].precision,
                    "recall": rows[0].recall,
                    "f1": rows[0].f1,
                }
            ]
        )

    rows = []
    for thr in np.linspace(s_min, s_max, n_steps):
        res = evaluate_threshold(y_true, scores, float(thr), name="sweep")
        rows.append(
            {
                "threshold": res.threshold,
                "precision": res.precision,
                "recall": res.recall,
                "f1": res.f1,
            }
        )

    return pd.DataFrame(rows)
