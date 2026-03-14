"""
Signal preprocessing utilities for IMU-based gait analysis.

Provides bandpass filtering, overlapping windowing, and sample-level
artifact rejection for 3-axis accelerometer data.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(
    acc: np.ndarray,
    sample_rate: int,
    lowcut: float = 0.5,
    highcut: float = 10.0,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter to each accelerometer axis.

    Uses second-order sections (SOS) representation and ``sosfiltfilt`` for
    zero-phase (forward-backward) filtering to avoid phase distortion.

    Parameters
    ----------
    acc : np.ndarray
        Input signal of shape ``(n_samples, n_axes)``.
    sample_rate : int
        Sampling rate of the signal in Hz.
    lowcut : float, optional
        Lower passband edge in Hz. Default is 0.5 Hz.
    highcut : float, optional
        Upper passband edge in Hz. Default is 10.0 Hz.

    Returns
    -------
    np.ndarray
        Filtered signal of shape ``(n_samples, n_axes)``, same dtype as input.

    Raises
    ------
    ValueError
        If ``highcut`` is greater than or equal to the Nyquist frequency
        (``sample_rate / 2``).
    """
    nyquist = sample_rate / 2.0
    if highcut >= nyquist:
        raise ValueError(
            f"highcut ({highcut} Hz) must be strictly less than the Nyquist "
            f"frequency ({nyquist} Hz) for sample_rate={sample_rate} Hz."
        )

    sos = butter(4, [lowcut / nyquist, highcut / nyquist], btype="band", output="sos")
    return sosfiltfilt(sos, acc, axis=0).astype(acc.dtype)


def window_signal(
    acc: np.ndarray,
    window_size: int,
    step_size: int,
) -> np.ndarray:
    """
    Segment a signal into overlapping fixed-length windows.

    Parameters
    ----------
    acc : np.ndarray
        Input signal of shape ``(n_samples, n_axes)``.
    window_size : int
        Number of samples per window.
    step_size : int
        Number of samples to advance between consecutive windows.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_windows, window_size, n_axes)`` where
        ``n_windows = floor((n_samples - window_size) / step_size) + 1``.
        Returns a view where possible; falls back to a copy for non-contiguous
        strides.

    Notes
    -----
    Trailing samples that do not fill a complete window are discarded.
    """
    n_samples, n_axes = acc.shape
    n_windows = (n_samples - window_size) // step_size + 1

    # Build a strided view over the sample axis to avoid explicit copying.
    shape = (n_windows, window_size, n_axes)
    byte = acc.strides[0]
    strides = (step_size * byte, byte, acc.strides[1])
    windows = np.lib.stride_tricks.as_strided(acc, shape=shape, strides=strides)

    # as_strided returns a view; copy only if the caller needs writeable data.
    return windows


def reject_artifacts(
    acc: np.ndarray,
    clip_threshold: float = 14.0,
) -> np.ndarray:
    """
    Produce a boolean mask identifying clean (non-artifact) samples.

    A sample is considered an artifact if any axis exceeds ``clip_threshold``
    in absolute value, which is the canonical indicator of accelerometer
    clipping or shock transients.

    Parameters
    ----------
    acc : np.ndarray
        Input signal of shape ``(n_samples, n_axes)``.
    clip_threshold : float, optional
        Absolute value threshold in g. Samples where ``|acc| >= threshold``
        on any axis are flagged. Default is 15.0 g.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(n_samples,)`` where ``True`` indicates a
        clean sample and ``False`` indicates an artifact.
    """
    return np.all(np.abs(acc) < clip_threshold, axis=1)
