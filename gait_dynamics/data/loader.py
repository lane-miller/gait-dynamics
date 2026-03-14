"""
Data loader for the Camargo et al. PhysioNet lower-limb biomechanics dataset.

Dataset reference
-----------------
Camargo, J., Ramanathan, A., Flanagan, W., & Young, A. (2021).
A comprehensive, open-source dataset of lower limb biomechanics in multiple
conditions of stairs, ramps, and level-ground ambulation.
Journal of Biomechanics, 119, 110320.
PhysioNet: https://physionet.org/content/lower-limb-biomechanics/1.0/

Directory structure convention
-------------------------------
::

    <data_dir>/
        AB01/
            AB01_<trial>.hea
            AB01_<trial>.dat
            ...
        AB02/
            ...

Each subject occupies a single subdirectory whose name is the subject
identifier (e.g. ``"AB01"``).  WFDB record files (``.hea`` / ``.dat``)
reside directly inside that subdirectory.  Accelerometer channels are
identified by the presence of ``"acc"`` (case-insensitive) in the WFDB
signal name field of the header file.
"""

from pathlib import Path

import numpy as np
import wfdb


def list_subjects(data_dir: str | Path) -> list[str]:
    """
    Return a sorted list of subject identifiers found in the data directory.

    A valid subject directory is any non-hidden subdirectory that contains
    at least one WFDB header file (``.hea``).

    Parameters
    ----------
    data_dir : str or Path
        Path to the root directory of the downloaded dataset.

    Returns
    -------
    list[str]
        Sorted list of subject identifier strings (directory names).

    Raises
    ------
    FileNotFoundError
        If ``data_dir`` does not exist.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: '{data_dir}'. "
            "Download the Camargo et al. PhysioNet dataset and point "
            "data_dir at its root."
        )

    return sorted(
        entry.name
        for entry in data_dir.iterdir()
        if entry.is_dir()
        and not entry.name.startswith(".")
        and any(entry.glob("*.hea"))
    )


def load_subject(
    data_dir: str | Path,
    subject_id: str,
    trial_id: str | None = None,
) -> dict:
    """
    Load IMU accelerometer data for a single subject.

    Reads a WFDB record from the subject's directory and extracts the three
    accelerometer channels identified by signal names containing ``"acc"``
    (case-insensitive).

    Parameters
    ----------
    data_dir : str or Path
        Path to the root directory of the downloaded dataset.
    subject_id : str
        Subject identifier matching a subdirectory name (e.g. ``"AB01"``).
    trial_id : str or None, optional
        WFDB record stem to load (e.g. ``"AB01_LG_R01"``).  When ``None``
        (default), the first record found alphabetically is loaded.

    Returns
    -------
    dict
        Dictionary with the following keys:

        ``subject_id`` : str
            The subject identifier passed in.
        ``acc`` : np.ndarray
            Accelerometer signal of shape ``(n_samples, 3)`` in physical
            units (g), ordered ``[ax, ay, az]`` as listed in the header.
        ``sample_rate`` : int
            Sampling rate of the record in Hz.

    Raises
    ------
    FileNotFoundError
        If the subject directory does not exist, if no WFDB records are
        found inside it, or if the requested ``trial_id`` is not present.
        The error message for a missing trial lists all available trial
        identifiers.
    ValueError
        If the number of accelerometer channels identified in the record
        is not exactly 3.
    """
    subject_dir = Path(data_dir) / subject_id

    if not subject_dir.exists():
        raise FileNotFoundError(
            f"Subject directory not found: '{subject_dir}'. "
            f"Verify that subject '{subject_id}' exists in '{data_dir}'."
        )

    header_files = sorted(subject_dir.glob("*.hea"))
    if not header_files:
        raise FileNotFoundError(
            f"No WFDB records (.hea) found in '{subject_dir}'."
        )

    # Build a stem → Path mapping once so lookup and error reporting share
    # the same source of truth.
    available: dict[str, Path] = {f.stem: f for f in header_files}

    if trial_id is None:
        # Alphabetically first record — same behaviour as before.
        record_path = header_files[0].with_suffix("")
    else:
        if trial_id not in available:
            raise FileNotFoundError(
                f"Trial '{trial_id}' not found for subject '{subject_id}'. "
                f"Available trials: {sorted(available)}"
            )
        record_path = available[trial_id].with_suffix("")

    # Strip the suffix — wfdb.rdrecord expects the bare record path.
    record = wfdb.rdrecord(str(record_path))

    # p_signal holds physical-unit values; fall back to digitised signal
    # if the header omits physical gain/offset information.
    signal: np.ndarray
    if record.p_signal is not None:
        signal = record.p_signal
    else:
        signal = record.d_signal.astype(np.float64)

    acc_indices = [
        i
        for i, name in enumerate(record.sig_name)
        if "acc" in name.lower()
    ]

    if len(acc_indices) != 3:
        raise ValueError(
            f"Expected exactly 3 accelerometer axes in record "
            f"'{record_path.name}', found {len(acc_indices)}. "
            f"Available signal names: {record.sig_name}"
        )

    return {
        "subject_id": subject_id,
        "acc": signal[:, acc_indices],
        "sample_rate": int(record.fs),
    }
