from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from sklearn.neural_network import MLPRegressor


DEFAULT_HIDDEN_LAYERS = (32, 16, 8, 16, 32)


@dataclass
class AEConfig:
    hidden_layer_sizes: tuple[int, ...] = DEFAULT_HIDDEN_LAYERS
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 1e-4
    batch_size: int = 256
    learning_rate_init: float = 1e-3
    max_iter: int = 60
    tol: float = 1e-4
    n_iter_no_change: int = 5
    early_stopping: bool = True
    validation_fraction: float = 0.1
    random_state: int = 42


def build_autoencoder(config: Optional[AEConfig] = None) -> MLPRegressor:
    cfg = config or AEConfig()

    model = MLPRegressor(
        hidden_layer_sizes=cfg.hidden_layer_sizes,
        activation=cfg.activation,
        solver=cfg.solver,
        alpha=cfg.alpha,
        batch_size=cfg.batch_size,
        learning_rate_init=cfg.learning_rate_init,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
        n_iter_no_change=cfg.n_iter_no_change,
        early_stopping=cfg.early_stopping,
        validation_fraction=cfg.validation_fraction,
        random_state=cfg.random_state,
    )
    return model


def fit_autoencoder(model: MLPRegressor, x_train: np.ndarray) -> MLPRegressor:
    model.fit(x_train, x_train)
    return model


def reconstruct(model: MLPRegressor, x: np.ndarray) -> np.ndarray:
    return model.predict(x)


def reconstruction_error_matrix(x_true: np.ndarray, x_recon: np.ndarray) -> np.ndarray:
    diff = x_true - x_recon
    return np.square(diff)


def sample_reconstruction_scores(x_true: np.ndarray, x_recon: np.ndarray) -> np.ndarray:
    err = reconstruction_error_matrix(x_true, x_recon)
    return np.mean(err, axis=1)


def feature_reconstruction_scores(x_true: np.ndarray, x_recon: np.ndarray) -> np.ndarray:
    return reconstruction_error_matrix(x_true, x_recon)


def top_k_feature_indices(
    feature_errors: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    if feature_errors.ndim != 2:
        raise ValueError("A feature_errors 2 dimenziós mátrix kell legyen.")
    if k <= 0:
        raise ValueError("A k értékének pozitívnak kell lennie.")

    k = min(k, feature_errors.shape[1])
    return np.argsort(-feature_errors, axis=1)[:, :k]


def summarize_top_k_feature_names(
    feature_errors: np.ndarray,
    feature_names: Iterable[str],
    k: int = 5,
) -> list[list[str]]:
    feature_names = list(feature_names)
    top_idx = top_k_feature_indices(feature_errors, k=k)

    out: list[list[str]] = []
    for row in top_idx:
        out.append([feature_names[i] for i in row])
    return out
