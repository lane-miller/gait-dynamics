"""
Feature extraction utilities for IMU-based gait analysis.

Provides cadence estimation, stride symmetry quantification, and spectral
feature computation from 3-axis accelerometer data.
"""

import numpy as np
from scipy.signal import find_peaks, welch

_PROMINENCE_SCALE = 0.3  # fraction of signal std used for peak prominence filtering


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
        prominence=np.std(vertical) * _PROMINENCE_SCALE,
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


# ── Feature registry ──────────────────────────────────────────────────────────

FEATURE_REGISTRY: dict[str, callable] = {
    "cadence": compute_cadence,
    "stride_symmetry": compute_stride_symmetry,
    "spectral": compute_spectral_features,
}


# ── Feature matrix builder ────────────────────────────────────────────────────

def _coerce_to_array(result) -> np.ndarray:
    """Convert a feature function's return value to a 1-D float64 array."""
    if isinstance(result, dict):
        return np.array(list(result.values()), dtype=np.float64)
    return np.array([result], dtype=np.float64)


def build_feature_matrix(
    windows: np.ndarray,
    sample_rate: int,
    feature_names: list[str],
) -> np.ndarray:
    """
    Extract features from each window and assemble a 2-D feature matrix.

    For every window the selected feature functions are called in order and
    their outputs are concatenated into a single row vector.  Scalar returns
    (``float``) contribute one column; ``dict`` returns contribute one column
    per value, in insertion order.

    Parameters
    ----------
    windows : np.ndarray
        Segmented signal of shape ``(n_windows, window_size, n_axes)`` as
        produced by :func:`~gait_dynamics.data.preprocess.window_signal`.
    sample_rate : int
        Sampling rate of the original signal in Hz, forwarded to each
        feature function.
    feature_names : list[str]
        Ordered list of keys from :data:`FEATURE_REGISTRY` selecting which
        features to compute.  The column order of the output matrix follows
        the order of this list.

    Returns
    -------
    np.ndarray
        Feature matrix of shape ``(n_windows, n_features)`` with dtype
        ``float64``.  ``n_features`` is the sum of the output widths of the
        selected functions (1 per scalar return, ``len(dict)`` per dict
        return).

    Raises
    ------
    ValueError
        If any name in ``feature_names`` is not a key in
        :data:`FEATURE_REGISTRY`.  All unknown names are reported together.

    Notes
    -----
    Column width contributed by each registry entry:

    * ``"cadence"`` → 1 (scalar ``float``)
    * ``"stride_symmetry"`` → 1 (scalar ``float``)
    * ``"spectral"`` → 2 (``dominant_frequency``, ``spectral_entropy``)

    Feature columns are ordered by ``feature_names`` list order, then by
    dict insertion order within each spectral feature. This ordering is
    stable in Python 3.7+ and must be consistent between training and
    inference runs.
    """
    unknown = [n for n in feature_names if n not in FEATURE_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown feature name(s): {unknown}. "
            f"Valid options are: {sorted(FEATURE_REGISTRY)}"
        )

    if len(windows) == 0:
        return np.empty((0, 0), dtype=np.float64)

    funcs = [FEATURE_REGISTRY[n] for n in feature_names]

    # Dry-run on the first window to determine n_features, then pre-allocate.
    first_parts = [_coerce_to_array(fn(windows[0], sample_rate)) for fn in funcs]
    n_features = sum(p.shape[0] for p in first_parts)
    out = np.empty((len(windows), n_features), dtype=np.float64)
    out[0] = np.concatenate(first_parts)

    for i in range(1, len(windows)):
        parts = [_coerce_to_array(fn(windows[i], sample_rate)) for fn in funcs]
        out[i] = np.concatenate(parts)

    return out
