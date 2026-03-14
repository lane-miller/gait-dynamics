"""
Unit tests for scripts.predict module.

joblib.load and load_subject are patched. bandpass_filter, window_signal, and
build_feature_matrix run for real on a zero-valued array — each degrades
gracefully to zero/constant output so no additional mocking is needed.
"""

import argparse
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scripts.predict import run_prediction

# ── Shared test data ──────────────────────────────────────────────────────────

# Predictions returned by the mock pipeline: 5 windows, classes 0 and 1.
_KNOWN_PREDICTIONS = np.array([0, 1, 0, 0, 1])

_FAKE_SUBJECT_DATA = {"acc": np.zeros((500, 3)), "sample_rate": 100}


def _args(model_path: str, **overrides) -> argparse.Namespace:
    """Return a Namespace with sensible defaults, optionally overridden."""
    defaults = dict(
        model_path=model_path,
        data_dir="/data",
        subject_id="AB01",
        trial_id="AB01_LG_R01",
        window_size=200,
        step_size=100,
        features=["cadence"],
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture
def mock_pipeline():
    """Sklearn pipeline stub whose predict() returns _KNOWN_PREDICTIONS."""
    pipeline = MagicMock()
    pipeline.predict.return_value = _KNOWN_PREDICTIONS
    return pipeline


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRunPrediction:

    def test_raises_fnf_for_missing_model_file(self, tmp_path):
        """Non-existent model_path must raise FileNotFoundError."""
        args = _args(model_path=str(tmp_path / "nonexistent.joblib"))
        with pytest.raises(FileNotFoundError, match="nonexistent.joblib"):
            run_prediction(args)

    def test_returns_dict_with_correct_keys(self, tmp_path, mock_pipeline):
        """Returned dict must contain all six documented keys."""
        model_file = tmp_path / "model.joblib"
        model_file.touch()

        with (
            patch("scripts.predict.joblib.load", return_value=mock_pipeline),
            patch("scripts.predict.load_subject", return_value=_FAKE_SUBJECT_DATA),
        ):
            result = run_prediction(_args(str(model_file)))

        assert set(result.keys()) == {
            "subject_id",
            "trial_id",
            "model_path",
            "n_windows",
            "predictions",
            "prediction_counts",
        }

    def test_prediction_counts_sum_to_n_windows(self, tmp_path, mock_pipeline):
        """Sum of all prediction_counts values must equal n_windows."""
        model_file = tmp_path / "model.joblib"
        model_file.touch()

        with (
            patch("scripts.predict.joblib.load", return_value=mock_pipeline),
            patch("scripts.predict.load_subject", return_value=_FAKE_SUBJECT_DATA),
        ):
            result = run_prediction(_args(str(model_file)))

        assert sum(result["prediction_counts"].values()) == result["n_windows"]
