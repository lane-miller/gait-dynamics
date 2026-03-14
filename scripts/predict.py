"""
CLI inference script for gait activity classification using a saved pipeline.

Loads a serialized scikit-learn pipeline produced by ``scripts/train.py``,
processes a single subject/trial through the same preprocessing and feature
extraction steps used during training, runs ``pipeline.predict``, and prints
a per-class prediction distribution to stdout.

Activity class labels
---------------------
0 — level_walking
1 — stair_ascent
2 — stair_descent
3 — ramp_ascent
4 — ramp_descent

Example
-------
.. code-block:: bash

    python scripts/predict.py \\
        --model-path models/AB01_xgb.joblib \\
        --data-dir data/raw \\
        --subject-id AB01 \\
        --trial-id AB01_LG_R01 \\
        --window-size 200 \\
        --step-size 100 \\
        --features cadence stride_symmetry spectral
"""

import argparse
from pathlib import Path

import joblib
import numpy as np

from gait_dynamics.data.loader import load_subject
from gait_dynamics.data.preprocess import bandpass_filter, window_signal
from gait_dynamics.features.extraction import FEATURE_REGISTRY, build_feature_matrix
from gait_dynamics.models.sklearn_pipeline import ACTIVITY_LABELS


def run_prediction(args: argparse.Namespace) -> dict:
    """
    Load a saved pipeline and run inference on a single subject trial.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.  Expected attributes:

        * ``model_path`` (str) — path to a ``.joblib`` pipeline file.
        * ``data_dir`` (str) — root directory of the dataset.
        * ``subject_id`` (str) — subject identifier (e.g. ``"AB01"``).
        * ``trial_id`` (str or None) — WFDB record stem; ``None`` loads the
          first trial alphabetically.
        * ``window_size`` (int) — samples per window.
        * ``step_size`` (int) — samples advanced between windows.
        * ``features`` (list[str]) — feature names to compute.

    Returns
    -------
    dict
        Dictionary with the following keys:

        ``subject_id`` : str
            The subject identifier passed in.
        ``trial_id`` : str or None
            The trial identifier passed in.
        ``model_path`` : str
            Absolute path to the loaded ``.joblib`` file.
        ``n_windows`` : int
            Total number of windows fed to the classifier.
        ``predictions`` : np.ndarray
            Integer class labels of shape ``(n_windows,)``.
        ``prediction_counts`` : dict
            Mapping of ``{class_label_int: window_count}`` for every class
            that appears at least once in ``predictions``.

    Raises
    ------
    FileNotFoundError
        If ``model_path`` does not point to an existing file.
    """
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: '{model_path}'. "
            "Train a pipeline first with scripts/train.py."
        )

    # ── 1. Load pipeline ──────────────────────────────────────────────────────
    print(f"[1/4] Loading pipeline from '{model_path}' …")
    pipeline = joblib.load(model_path)

    # ── 2. Load and preprocess subject data ───────────────────────────────────
    print(
        f"[2/4] Loading subject '{args.subject_id}' "
        f"(trial: {args.trial_id or 'first alphabetically'}) …"
    )
    subject_data = load_subject(
        args.data_dir,
        args.subject_id,
        trial_id=args.trial_id,
    )
    acc: np.ndarray = subject_data["acc"]
    sample_rate: int = subject_data["sample_rate"]
    print(f"      acc shape: {acc.shape}, sample_rate: {sample_rate} Hz")

    acc_filtered = bandpass_filter(acc, sample_rate)
    windows = window_signal(acc_filtered, args.window_size, args.step_size)
    print(
        f"      {windows.shape[0]} windows "
        f"(window={args.window_size}, step={args.step_size})"
    )

    # ── 3. Extract features ───────────────────────────────────────────────────
    print(f"[3/4] Extracting features: {args.features} …")
    X = build_feature_matrix(windows, sample_rate, args.features)
    print(f"      Feature matrix shape: {X.shape}")

    # ── 4. Predict ────────────────────────────────────────────────────────────
    print("[4/4] Running inference …")
    predictions: np.ndarray = pipeline.predict(X)
    n_windows = int(len(predictions))

    unique, counts = np.unique(predictions, return_counts=True)
    prediction_counts: dict[int, int] = {
        int(label): int(count) for label, count in zip(unique, counts)
    }

    return {
        "subject_id": args.subject_id,
        "trial_id": args.trial_id,
        "model_path": str(model_path.resolve()),
        "n_windows": n_windows,
        "predictions": predictions,
        "prediction_counts": prediction_counts,
    }


def _print_summary(result: dict) -> None:
    """
    Print a human-readable prediction summary to stdout.

    Parameters
    ----------
    result : dict
        Return value of :func:`run_prediction`.
    """
    print("\n── Prediction Summary ───────────────────────────────────")
    print(f"  Subject   : {result['subject_id']}")
    print(f"  Trial     : {result['trial_id'] or '(first alphabetically)'}")
    print(f"  Model     : {result['model_path']}")
    print(f"  Windows   : {result['n_windows']}")
    print("\n  Predicted activity distribution:")

    counts = result["prediction_counts"]
    for class_idx, label in enumerate(ACTIVITY_LABELS):
        count = counts.get(class_idx, 0)
        pct = count / result["n_windows"] * 100 if result["n_windows"] > 0 else 0.0
        bar = "█" * int(pct / 2)
        print(f"    [{class_idx}] {label:<18}  {count:>4} windows  ({pct:5.1f}%)  {bar}")


def main() -> None:
    """Parse CLI arguments and run inference."""
    parser = argparse.ArgumentParser(
        prog="predict",
        description="Run gait activity inference using a saved pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        required=True,
        metavar="PATH",
        help="Path to a .joblib pipeline file produced by scripts/train.py.",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        metavar="PATH",
        help="Root directory of the Camargo et al. PhysioNet dataset.",
    )
    parser.add_argument(
        "--subject-id",
        required=True,
        metavar="ID",
        help="Subject identifier matching a subdirectory name (e.g. 'AB01').",
    )
    parser.add_argument(
        "--trial-id",
        default=None,
        metavar="ID",
        help="WFDB record stem to load (e.g. 'AB01_LG_R01'). "
             "Defaults to the first record found alphabetically.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=200,
        metavar="N",
        help="Number of samples per window.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=100,
        metavar="N",
        help="Number of samples to advance between consecutive windows.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=["cadence", "stride_symmetry", "spectral"],
        choices=sorted(FEATURE_REGISTRY),
        metavar="NAME",
        help=(
            "Feature names to compute (space-separated). "
            f"Valid choices: {sorted(FEATURE_REGISTRY)}."
        ),
    )

    args = parser.parse_args()
    result = run_prediction(args)
    _print_summary(result)


if __name__ == "__main__":
    main()
