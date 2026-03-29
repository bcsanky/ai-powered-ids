from __future__ import annotations

import numpy as np

from ml.src.autoencoder import (
    AEConfig,
    build_autoencoder,
    feature_reconstruction_scores,
    fit_autoencoder,
    reconstruct,
    sample_reconstruction_scores,
    summarize_top_k_feature_names,
)


def test_autoencoder_can_fit_and_reconstruct():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(50, 10)).astype(np.float32)

    config = AEConfig(
        hidden_layer_sizes=(16, 8, 16),
        max_iter=20,
        batch_size=16,
        random_state=42,
    )
    model = build_autoencoder(config)
    fit_autoencoder(model, x)

    x_recon = reconstruct(model, x)

    assert x_recon.shape == x.shape


def test_sample_and_feature_scores_have_expected_shapes():
    rng = np.random.default_rng(42)
    x_true = rng.normal(size=(20, 7)).astype(np.float32)
    x_recon = rng.normal(size=(20, 7)).astype(np.float32)

    feature_scores = feature_reconstruction_scores(x_true, x_recon)
    sample_scores = sample_reconstruction_scores(x_true, x_recon)

    assert feature_scores.shape == (20, 7)
    assert sample_scores.shape == (20,)


def test_top_k_feature_name_summary():
    feature_errors = np.array(
        [
            [0.1, 0.9, 0.2, 0.7],
            [0.8, 0.1, 0.6, 0.2],
        ],
        dtype=np.float32,
    )
    feature_names = ["f1", "f2", "f3", "f4"]

    top_names = summarize_top_k_feature_names(feature_errors, feature_names, k=2)

    assert len(top_names) == 2
    assert top_names[0] == ["f2", "f4"]
    assert top_names[1] == ["f1", "f3"]
