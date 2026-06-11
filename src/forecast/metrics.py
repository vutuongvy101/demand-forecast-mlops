"""Forecast evaluation metrics: RMSSE, MASE, sMAPE."""

from __future__ import annotations

import numpy as np


def rmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
) -> float:
    """Root Mean Squared Scaled Error (M5 official metric basis)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    if len(y_true) == 0:
        return np.nan

    scale = np.mean((y_train[1:] - y_train[:-1]) ** 2)
    if scale == 0:
        scale = 1.0

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2) / scale))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    if len(y_true) == 0:
        return np.nan

    if len(y_train) <= seasonality:
        scale = np.mean(np.abs(np.diff(y_train))) or 1.0
    else:
        scale = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality])) or 1.0

    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE — safe for intermittent series (unlike plain MAPE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        return np.nan

    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not mask.any():
        return 0.0

    return float(100.0 * np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]))


def compute_series_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, float]:
    return {
        "rmsse": rmsse(y_true, y_pred, y_train),
        "mase": mase(y_true, y_pred, y_train),
        "smape": smape(y_true, y_pred),
    }


