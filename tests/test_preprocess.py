"""
Unit tests for gait_dynamics.data.preprocess module.
"""

import numpy as np
import pytest
from tests.conftest import make_synthetic_acc, make_artifact_signal


class TestBandpassFilter:

    def test_output_shape_matches_input(self, synthetic_acc, sample_rate):
        """Filter should not change the shape of the input array."""
        pass

    def test_attenuates_noise_above_cutoff(self, synthetic_acc, sample_rate):
        """Signal energy above cutoff frequency should be reduced after filtering."""
        pass

    def test_raises_on_invalid_cutoff(self, synthetic_acc, sample_rate):
        """Should raise ValueError if cutoff exceeds Nyquist frequency."""
        pass


class TestWindowing:

    def test_output_shape_is_correct(self, synthetic_acc, sample_rate):
        """Windowed output shape should be (n_windows, window_size, n_axes)
        where n_windows = floor((n_samples - window_size) / step_size) + 1."""
        pass


class TestArtifactRejection:

    def test_flags_clipping_artifact(self, sample_rate):
        """Clipping artifact should be detected and flagged."""
        clean = make_synthetic_acc()
        artifact = make_artifact_signal(clean, time_pct=0.2, duration_pct=0.05)
        pass

    def test_clean_signal_not_flagged(self, synthetic_acc):
        """Clean signal should return no artifact flags."""
        pass