"""
Unit tests for gait_dynamics.models.sklearn_pipeline module.

Synthetic data is module-level: X is (100, 5) float, y is (100,) integer
labels with 5 balanced classes of 20 samples each. No real IMU data needed.
"""

import numpy as np
import joblib
import pytest
from sklearn.pipeline import Pipeline

from gait_dynamics.models.sklearn_pipeline import (
    build_pipeline,
    save_pipeline,
    train_evaluate,
)

# ── Shared synthetic dataset ──────────────────────────────────────────────────
# Module-level constants avoid re-generating data for every test and
# ensure all tests share identical inputs for cross-test comparability.
_X = np.random.default_rng(0).standard_normal((100, 5))
_y = np.repeat(np.arange(5), 20)          # 5 classes × 20 samples each


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def lr_report():
    """Evaluation report from a fresh 'lr' pipeline on the synthetic dataset."""
    return train_evaluate(build_pipeline("lr"), _X, _y)


@pytest.fixture
def fitted_lr_pipeline():
    """'lr' pipeline fitted on the full synthetic dataset."""
    pipeline = build_pipeline("lr")
    pipeline.fit(_X, _y)
    return pipeline


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBuildPipeline:

    @pytest.mark.parametrize("model_type", ["xgb", "svm", "lr"])
    def test_returns_pipeline_instance_for_valid_model_type(self, model_type):
        """Each supported model_type must produce a sklearn Pipeline."""
        assert isinstance(build_pipeline(model_type), Pipeline)

    def test_raises_value_error_for_unsupported_model_type(self):
        """Unsupported model_type string must raise ValueError."""
        with pytest.raises(ValueError, match="rf"):
            build_pipeline("rf")

    @pytest.mark.parametrize("model_type", ["xgb", "svm", "lr"])
    def test_last_step_name_matches_model_type(self, model_type):
        """The last step name in the pipeline must equal the model_type passed in."""
        assert build_pipeline(model_type).steps[-1][0] == model_type


class TestTrainEvaluate:

    def test_returns_dict_with_correct_keys(self, lr_report):
        """Returned dict must contain exactly the five documented keys."""
        assert set(lr_report.keys()) == {
            "model_type",
            "accuracy",
            "classification_report",
            "n_train",
            "n_test",
        }

    def test_accuracy_is_float_in_unit_interval(self, lr_report):
        """Accuracy must be a plain Python float within [0.0, 1.0]."""
        assert isinstance(lr_report["accuracy"], float)
        assert 0.0 <= lr_report["accuracy"] <= 1.0

    def test_n_train_plus_n_test_equals_total_samples(self, lr_report):
        """n_train and n_test must partition the full input without overlap."""
        assert lr_report["n_train"] + lr_report["n_test"] == len(_y)


class TestSavePipeline:

    def test_saved_file_exists_at_path(self, fitted_lr_pipeline, tmp_path):
        """File must exist at the given path after save_pipeline returns.

        Uses a nested subdirectory to exercise the parents=True mkdir logic.
        """
        dest = tmp_path / "checkpoints" / "gait_lr.joblib"
        save_pipeline(fitted_lr_pipeline, dest)
        assert dest.exists()

    def test_reloaded_pipeline_makes_identical_predictions(self, fitted_lr_pipeline, tmp_path):
        """Pipeline reloaded via joblib.load must predict identically to the original."""
        dest = tmp_path / "gait_lr.joblib"
        save_pipeline(fitted_lr_pipeline, dest)
        reloaded = joblib.load(dest)
        np.testing.assert_array_equal(
            fitted_lr_pipeline.predict(_X),
            reloaded.predict(_X),
        )
