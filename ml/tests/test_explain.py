from __future__ import annotations

import numpy as np
import pandas as pd

from ml.src.explain import aggregate_feature_errors, top_k_group_names

def test_aggregate_feature_errors_groups_protocol_columns():
    feature_names = ["destination_port", "flow_duration", "protocol_tcp", "protocol_udp"]
    feature_error_matrix = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ],
        dtype=np.float32,
    )

    grouped = aggregate_feature_errors(feature_error_matrix, feature_names)

    assert list(grouped.columns) == ["destination_port", "flow_duration", "protocol"]
    assert np.isclose(grouped.iloc[0]["protocol"], 0.7, atol=1e-7, rtol=0)
    assert np.isclose(grouped.iloc[1]["protocol"], 1.5, atol=1e-7, rtol=0)


def test_top_k_group_names_returns_sorted_names():
    grouped = pd.DataFrame(
        [
            {"a": 0.1, "b": 0.9, "c": 0.4},
            {"a": 0.8, "b": 0.2, "c": 0.6},
        ]
    )

    top_names = top_k_group_names(grouped, k=2)

    assert top_names[0] == ["b", "c"]
    assert top_names[1] == ["a", "c"]
