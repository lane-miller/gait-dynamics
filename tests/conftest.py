"""
Shared pytest fixtures and signal generation helpers for gait_dynamics test suite.
Generates synthetic IMU signals to avoid dependency on real dataset.
"""

import numpy as np
import pytest


# ── Helper functions (callable directly in tests with custom parameters) ──

def make_synthetic_acc(
    sample_rate: int = 100,
    duration: float = 10.0,
    step_freq: float = 2.0,
    snr_db: float = 20.0,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic 3-axis accelerometer signal simulating walking.

    Parameters
    ----------
    sample_rate : int
        Sampling rate in Hz.
    duration : float
        Signal duration in seconds.
    step_freq : float
        Simulated step frequency in Hz.
    snr_db : float
        Signal-to-noise ratio in dB. Higher = cleaner signal.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, 3) representing [ax, ay, az].
    """
    rng = np.random.default_rng(random_seed)
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n)

    # Signal amplitudes per axis
    ax_clean = 0.3 * np.sin(2 * np.pi * step_freq * t)
    ay_clean = 0.1 * np.sin(2 * np.pi * step_freq * t + np.pi / 4)
    az_clean = 0.2 * np.sin(2 * np.pi * (step_freq / 2) * t)

    # Compute noise std from SNR: SNR_db = 20*log10(signal_rms / noise_std)
    def add_noise(clean: np.ndarray) -> np.ndarray:
        signal_rms = np.sqrt(np.mean(clean ** 2))
        noise_std = signal_rms / (10 ** (snr_db / 20))
        return clean + rng.normal(0, noise_std, size=clean.shape)

    return np.column_stack([add_noise(ax_clean), add_noise(ay_clean), add_noise(az_clean)])


def make_artifact_signal(
    base_signal: np.ndarray,
    sample_rate: int = 100,
    time_pct: float = 0.2,
    duration_pct: float = 0.05,
    clip_value: float = 16.0,
) -> np.ndarray:
    """
    Inject a clipping artifact into a copy of base_signal.

    Parameters
    ----------
    base_signal : np.ndarray
        Clean input signal of shape (n_samples, 3).
    sample_rate : int
        Sampling rate in Hz (unused directly, kept for future use).
    time_pct : float
        Artifact start position as fraction of signal length (0.0 to 1.0).
    duration_pct : float
        Artifact duration as fraction of signal length (0.0 to 1.0).
    clip_value : float
        Saturated value to inject (typical MEMS accelerometer clip at ~16g).

    Returns
    -------
    np.ndarray
        Signal with artifact injected, same shape as base_signal.
    """
    signal = base_signal.copy()
    n = signal.shape[0]
    start = int(time_pct * n)
    end = int((time_pct + duration_pct) * n)
    signal[start:end, 0] = clip_value
    return signal


# ── Fixtures (used automatically by pytest via parameter injection) ────────

@pytest.fixture
def sample_rate() -> int:
    """Standard IMU sample rate in Hz."""
    return 100


@pytest.fixture
def duration() -> float:
    """Signal duration in seconds."""
    return 10.0


@pytest.fixture
def synthetic_acc(sample_rate, duration) -> np.ndarray:
    """Clean synthetic 3-axis accelerometer signal at default SNR (20 dB)."""
    return make_synthetic_acc(sample_rate=sample_rate, duration=duration)


@pytest.fixture
def synthetic_acc_with_artifact(synthetic_acc, sample_rate) -> np.ndarray:
    """Synthetic accelerometer signal with a clipping artifact injected at 20% mark."""
    return make_artifact_signal(synthetic_acc, sample_rate=sample_rate)