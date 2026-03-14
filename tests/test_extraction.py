"""
Unit tests for gait_dynamics.features.extraction module.
"""

import numpy as np
import pytest
from tests.conftest import make_synthetic_acc
from gait_dynamics.features.extraction import (
    FEATURE_REGISTRY,
    build_feature_matrix,
    compute_cadence,
    compute_spectral_features,
    compute_stride_symmetry,
)


class TestComputeCadence:

    def test_known_step_frequency_yields_correct_cadence(self, synthetic_acc, sample_rate):
        """2 Hz step signal over 10 s should produce cadence of 120 steps/min."""
        # synthetic_acc uses step_freq=2.0 Hz → 2.0 * 60 = 120 steps/min
        result = compute_cadence(synthetic_acc, sample_rate)
        assert result == pytest.approx(120.0, abs=5.0)

    def test_cadence_scales_with_step_frequency(self, sample_rate):
        """Cadence from a 3 Hz signal should exceed cadence from a 1.5 Hz signal."""
        slow = make_synthetic_acc(sample_rate=sample_rate, step_freq=1.5, snr_db=40.0)
        fast = make_synthetic_acc(sample_rate=sample_rate, step_freq=3.0, snr_db=40.0)
        assert compute_cadence(fast, sample_rate) > compute_cadence(slow, sample_rate)


class TestComputeStrideSymmetry:

    def test_clean_signal_returns_high_symmetry(self, sample_rate):
        """Near-noiseless symmetric signal should yield symmetry index above 0.85."""
        acc = make_synthetic_acc(sample_rate=sample_rate, snr_db=60.0)
        result = compute_stride_symmetry(acc, sample_rate)
        assert np.isfinite(result)        
        assert np.isclose(result, result, atol=1e-6) and 0.0 <= result <= 1.0
        assert result > 0.85

    def test_output_is_scalar_float(self, synthetic_acc, sample_rate):
        """Return value must be a plain Python float, not an array."""
        result = compute_stride_symmetry(synthetic_acc, sample_rate)
        assert isinstance(result, float)


class TestComputeSpectralFeatures:

    def test_dominant_frequency_matches_step_frequency(self, synthetic_acc, sample_rate):
        """Dominant frequency in the PSD should resolve to 2.0 Hz (±0.1 Hz)."""
        result = compute_spectral_features(synthetic_acc, sample_rate)
        assert result["dominant_frequency"] == pytest.approx(2.0, abs=0.1)

    def test_spectral_entropy_is_bounded(self, synthetic_acc, sample_rate):
        """Spectral entropy of a narrow-band signal must lie within [0, 1]."""
        result = compute_spectral_features(synthetic_acc, sample_rate)
        assert 0.0 <= result["spectral_entropy"] <= 1.0


class TestBuildFeatureMatrix:

    def test_output_shape_is_n_windows_by_n_features(self, synthetic_acc, sample_rate):
        """All-feature matrix from one window must have shape (1, 4).

        Column count: cadence→1, stride_symmetry→1, spectral→2 = 4 total.
        """
        windows = np.expand_dims(synthetic_acc, 0)   # (1, 1000, 3)
        result = build_feature_matrix(windows, sample_rate, list(FEATURE_REGISTRY))
        assert result.shape == (1, 4)

    def test_raises_value_error_for_unknown_feature_name(self, synthetic_acc, sample_rate):
        """Unregistered feature name must raise ValueError naming the bad key."""
        windows = np.expand_dims(synthetic_acc, 0)
        with pytest.raises(ValueError, match="unknown_feat"):
            build_feature_matrix(windows, sample_rate, ["unknown_feat"])

    def test_all_features_has_more_columns_than_single_feature(self, synthetic_acc, sample_rate):
        """Selecting all registry features must yield more columns than cadence alone."""
        windows = np.expand_dims(synthetic_acc, 0)
        all_cols = build_feature_matrix(windows, sample_rate, list(FEATURE_REGISTRY)).shape[1]
        one_col = build_feature_matrix(windows, sample_rate, ["cadence"]).shape[1]
        assert all_cols > one_col
