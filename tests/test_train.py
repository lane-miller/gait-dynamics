"""
Unit tests for scripts.train module.

Filesystem, wfdb, sqlite3, and sklearn training calls are patched.
Preprocessing and feature extraction (bandpass_filter, window_signal,
build_feature_matrix) run for real on zero-valued arrays: each degrades
gracefully to zero/constant output, so no mocking is needed there.
"""

import argparse
from unittest.mock import patch

import numpy as np
import pytest

from scripts.train import _subjects_key, run_training


# ── Shared test data ──────────────────────────────────────────────────────────

def _args(**overrides) -> argparse.Namespace:
    """Return a Namespace with sensible defaults, optionally overridden."""
    defaults = dict(
        data_dir="/data",
        subjects=["AB01"],
        model_type="lr",
        window_size=200,
        step_size=100,
        features=["cadence"],
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


_FAKE_SUBJECT_DATA = {"acc": np.zeros((500, 3)), "sample_rate": 100}

_FAKE_EVAL = {
    "model_type": "lr",
    "accuracy": 0.90,
    "classification_report": "",
    "n_train": 80,
    "n_test": 20,
}


# ── TestSubjectsKey ───────────────────────────────────────────────────────────

class TestSubjectsKey:

    def test_returns_all_for_all_sentinel(self):
        """Passing ['all'] as subjects_arg must return the string 'all'."""
        assert _subjects_key(["all"], ["AB01", "AB02"]) == "all"

    def test_returns_subject_id_for_single_subject(self):
        """A single resolved subject must return its ID verbatim."""
        assert _subjects_key(["AB01"], ["AB01"]) == "AB01"

    def test_returns_multi_string_for_multiple_subjects(self):
        """Two or more resolved subjects must return 'multi_N_subjects'."""
        assert _subjects_key(["AB01", "AB02"], ["AB01", "AB02"]) == "multi_2_subjects"


# ── TestRunTraining ───────────────────────────────────────────────────────────

class TestRunTraining:

    def test_calls_list_subjects_when_subjects_is_all(self):
        """list_subjects must be called with data_dir when subjects=['all']."""
        with patch("scripts.train.list_subjects", return_value=[]) as mock_ls:
            with pytest.raises(RuntimeError):
                run_training(_args(subjects=["all"]))
        mock_ls.assert_called_once_with("/data")

    def test_raises_runtime_error_when_no_windows_produced(self):
        """Every subject having zero trials must raise RuntimeError."""
        with patch("scripts.train._list_trials", return_value=[]):
            with pytest.raises(RuntimeError, match="No windows"):
                run_training(_args())

    def test_returns_dict_containing_model_path(self):
        """Successful run must return a dict that includes a 'model_path' key."""
        with (
            patch("scripts.train._list_trials", return_value=["AB01_LG_R01"]),
            patch("scripts.train.load_subject", return_value=_FAKE_SUBJECT_DATA),
            patch("scripts.train.train_evaluate", return_value=_FAKE_EVAL),
            patch("scripts.train.save_pipeline"),
            patch("scripts.train._init_db"),
            patch("scripts.train._log_run"),
        ):
            result = run_training(_args())

        assert "model_path" in result
