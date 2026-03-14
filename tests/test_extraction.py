"""
Unit tests for gait_dynamics.features.extraction module.
"""

import numpy as np
import pytest
from tests.conftest import make_synthetic_acc
from gait_dynamics.features.extraction import (
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
