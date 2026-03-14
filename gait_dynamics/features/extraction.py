"""
Feature extraction utilities for IMU-based gait analysis.

Provides cadence estimation, stride symmetry quantification, and spectral
feature computation from 3-axis accelerometer data.
"""

import numpy as np
from scipy.signal import find_peaks, welch


def compute_cadence(acc: np.ndarray, sample_rate: int) -> float:
    """
    Estimate walking or running cadence from the vertical accelerometer axis.

    Cadence is derived from the mean inter-peak interval of the vertical
    (axis-0) acceleration after DC removal. Peak prominence filtering is
    applied to suppress noise-induced false detections.

    Parameters
    ----------
    acc : np.ndarray
        3-axis accelerometer signal of shape ``(n_samples, n_axes)``.
        Axis 0 (first column) is treated as the vertical axis.
    sample_rate : int
        Sampling rate of the signal in Hz.

    Returns
    -------
    float
        Estimated cadence in steps per minute. Returns ``0.0`` if fewer
        than two peaks can be detected.

    Notes
    -----
    The minimum inter-peak distance is set to 0.2 s, permitting detection
    of step frequencies up to 5 Hz (300 steps/min), which covers sprinting.

    Axis convention assumes column 0 is the vertical (superior-inferior) axis.
    Verify axis assignment against dataset documentation before use with
    real IMU data, as conventions vary by sensor placement and manufacturer.
    """
    vertical = acc[:, 0].copy()
    vertical -= vertical.mean()

    # Minimum 0.2 s between peaks → supports up to 5 Hz / 300 steps·min⁻¹.
    min_distance = max(1, int(sample_rate * 0.2))

    peaks, _ = find_peaks(
        vertical,
        distance=min_distance,
        prominence=np.std(vertical) * 0.3,
    )

    if len(peaks) < 2:
        return 0.0

    mean_interval = float(np.mean(np.diff(peaks)))
    return float(sample_rate / mean_interval * 60.0)


def compute_stride_symmetry(acc: np.ndarray, sample_rate: int) -> float:
    """
    Compute a stride symmetry index from the autocorrelation of the vertical axis.

    The symmetry index is the normalized autocorrelation evaluated at the
    estimated stride period (twice the step period). A value of 1.0 indicates
    perfectly symmetric gait; 0.0 indicates complete asymmetry.

    Parameters
    ----------
    acc : np.ndarray
        3-axis accelerometer signal of shape ``(n_samples, n_axes)``.
        Axis 0 (first column) is treated as the vertical axis.
    sample_rate : int
        Sampling rate of the signal in Hz.

    Returns
    -------
    float
        Symmetry index in the range ``[0.0, 1.0]``.

    Notes
    -----
    The step period is located as the first prominent peak of the normalized
    autocorrelation in the range [0.25 s, 1.0 s], corresponding to step
    frequencies between 1 Hz and 4 Hz. The stride period is then taken as
    twice the step period.

    Step frequency search range of 1–4 Hz covers typical walking (1.6–2.2 Hz)
    and running (2.5–3.5 Hz). Signals outside this range (e.g. slow shuffling,
    sprinting) may require adjusted bounds passed as parameters in a future
    revision.
    """
    vertical = acc[:, 0].copy()
    vertical -= vertical.mean()

    # Full normalized autocorrelation; keep non-negative lags only.
    r_full = np.correlate(vertical, vertical, mode="full")
    r = r_full[len(r_full) // 2 :]
    zero_lag = r[0]
    if zero_lag == 0.0:
        return 0.0
    r = r / zero_lag

    # Search for the step-period peak between 0.25 s and 1.0 s.
    lag_min = max(1, int(sample_rate * 0.25))
    lag_max = int(sample_rate * 1.0)
    search_window = r[lag_min : lag_max + 1]

    peaks, _ = find_peaks(search_window)

    if len(peaks) == 0:
        # Fall back: use the lag with maximum autocorrelation in the window.
        t_step = int(lag_min + np.argmax(search_window))
    else:
        t_step = int(lag_min + peaks[0])

    t_stride = 2 * t_step
    if t_stride >= len(r):
        return float(np.clip(r[t_step], 0.0, 1.0))

    return float(np.clip(r[t_stride], 0.0, 1.0))


def compute_spectral_features(acc: np.ndarray, sample_rate: int) -> dict:
    """
    Compute power-spectral-density features of the vertical accelerometer axis.

    The PSD is estimated via Welch's method. Dominant frequency is the
    frequency bin with maximum power. Spectral entropy quantifies the
    flatness of the PSD distribution and is normalized to ``[0, 1]``.

    Parameters
    ----------
    acc : np.ndarray
        3-axis accelerometer signal of shape ``(n_samples, n_axes)``.
        Axis 0 (first column) is treated as the vertical axis.
    sample_rate : int
        Sampling rate of the signal in Hz.

    Returns
    -------
    dict
        Dictionary with the following keys:

        ``dominant_frequency`` : float
            Frequency (Hz) of the peak in the PSD.
        ``spectral_entropy`` : float
            Normalized Shannon entropy of the PSD in ``[0, 1]``. A value
            close to 0 indicates a narrow-band (rhythmic) signal; close to
            1 indicates a broadband (irregular) signal.

    Notes
    -----
    Welch's segment length is set to ``min(n_samples, 256)`` samples.
    Spectral entropy is computed as
    ``H = -sum(p * log2(p)) / log2(N)`` where ``p`` is the probability
    mass function derived from the PSD and ``N`` is the number of
    frequency bins.
    """
    vertical = acc[:, 0]
    n_samples = len(vertical)
    nperseg = min(n_samples, 256)

    freqs, psd = welch(vertical, fs=sample_rate, nperseg=nperseg)

    dominant_frequency = float(freqs[np.argmax(psd)])

    # Normalize PSD to a probability distribution, then compute Shannon entropy.
    psd_sum = psd.sum()
    if psd_sum == 0.0:
        return {"dominant_frequency": dominant_frequency, "spectral_entropy": 0.0}

    p = psd / psd_sum
    # Small epsilon prevents log(0); does not materially affect entropy for real PSDs.
    raw_entropy = float(-np.sum(p * np.log2(p + 1e-12)))
    spectral_entropy = float(np.clip(raw_entropy / np.log2(len(p)), 0.0, 1.0))

    return {
        "dominant_frequency": dominant_frequency,
        "spectral_entropy": spectral_entropy,
    }
