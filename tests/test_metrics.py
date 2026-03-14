"""
Unit tests for gait_dynamics.evaluation.metrics module.
"""

import math

import numpy as np
import pytest
from gait_dynamics.evaluation.metrics import compute_report, compute_rmse


class TestComputeRmse:

    def test_perfect_prediction_returns_zero(self):
        """RMSE of identical arrays must be exactly 0.0."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert compute_rmse(y, y) == 0.0

    def test_known_error_returns_exact_value(self):
        """[0, 0, 0] vs [3, 4, 0]: MSE = (9+16+0)/3 = 25/3, RMSE = sqrt(25/3)."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([3.0, 4.0, 0.0])
        expected = math.sqrt(25.0 / 3.0)
        assert compute_rmse(y_true, y_pred) == pytest.approx(expected)

    def test_return_type_is_float(self):
        """Return value must be a plain Python float, not a NumPy scalar."""
        result = compute_rmse(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
        assert isinstance(result, float)


class TestComputeReport:

    def test_model_name_is_preserved_in_output(self):
        """The model name passed in must appear verbatim in the returned dict."""
        y = np.array([1.0, 2.0, 3.0])
        result = compute_report(y, y, model_name="GRU_v1")
        assert result["model_name"] == "GRU_v1"

    def test_rmse_in_report_matches_standalone_function(self):
        """RMSE entry in the report must equal compute_rmse called on the same arrays."""
        y_true = np.array([2.0, 4.0, 6.0])
        y_pred = np.array([2.5, 3.5, 6.5])
        report = compute_report(y_true, y_pred, model_name="test")
        assert report["rmse"] == pytest.approx(compute_rmse(y_true, y_pred))

    def test_returns_dict(self):
        """Output must be a dict."""
        y = np.array([1.0, 2.0, 3.0])
        result = compute_report(y, y, model_name="any")
        assert isinstance(result, dict)
