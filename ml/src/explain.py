from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def grouped_feature_names(feature_names: Iterable[str]) -> list[str]:
    groups = []
    for name in feature_names:
        if name.startswith("protocol_"):
            groups.append("protocol")
        else:
            groups.append(name)
    return groups


def aggregate_feature_errors(
    feature_error_matrix: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    if feature_error_matrix.ndim != 2:
        raise ValueError("A feature_error_matrix 2 dimenziós kell legyen.")

    if feature_error_matrix.shape[1] != len(feature_names):
        raise ValueError(
            "A feature_error_matrix oszlopszáma és a feature_names hossza nem egyezik."
        )

    grouped_names = grouped_feature_names(feature_names)

    df = pd.DataFrame(feature_error_matrix, columns=feature_names)
    grouped: dict[str, list[str]] = {}

    for original_name, group_name in zip(feature_names, grouped_names):
        grouped.setdefault(group_name, []).append(original_name)

    out = pd.DataFrame(index=df.index)
    for group_name, cols in grouped.items():
        out[group_name] = df[cols].sum(axis=1)

    return out


def top_k_group_names(
    grouped_error_df: pd.DataFrame,
    k: int = 5,
) -> list[list[str]]:
    if k <= 0:
        raise ValueError("A k legyen pozitív.")

    out = []
    for _, row in grouped_error_df.iterrows():
        top_names = row.sort_values(ascending=False).head(k).index.tolist()
        out.append(top_names)
    return out
