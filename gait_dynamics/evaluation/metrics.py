"""
Evaluation metrics for gait analysis ML pipeline outputs.
"""

from datetime import datetime

import numpy as np


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute root mean squared error between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth target values of shape ``(n_samples,)`` or
        ``(n_samples, n_outputs)``.
    y_pred : np.ndarray
        Predicted values; must be the same shape as ``y_true``.

    Returns
    -------
    float
        RMSE as a plain Python float.
    """
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def compute_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> dict:
    """
    Build a summary evaluation report for a single model run.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth target values of shape ``(n_samples,)`` or
        ``(n_samples, n_outputs)``.
    y_pred : np.ndarray
        Predicted values; must be the same shape as ``y_true``.
    model_name : str
        Identifier for the model being evaluated.

    Returns
    -------
    dict
        Dictionary with the following keys:

        ``model_name`` : str
            The model identifier passed in.
        ``rmse`` : float
            Root mean squared error computed via :func:`compute_rmse`.
        ``n_samples`` : int
            Number of samples (first dimension of ``y_true``).
        ``timestamp`` : str
            ISO 8601 timestamp string of when the report was generated,
            e.g. ``"2026-03-14T09:15:00.123456"``.
    """
    y_true = np.asarray(y_true)
    return {
        "model_name": model_name,
        "rmse": compute_rmse(y_true, y_pred),
        "n_samples": int(y_true.shape[0]),
        "timestamp": datetime.now().isoformat(),
    }
