"""
CLI training script for gait activity classification using IMU-derived features.

Runs the full pipeline: load → filter → window → extract features → train →
evaluate → save model → log run to SQLite.

Subjects are selected with ``--subjects``.  Pass ``all`` to include every
subject found in ``--data-dir``, or pass one or more explicit IDs.  All WFDB
records (trials) found in each subject directory are loaded and their feature
matrices are concatenated before training.

Examples
--------
Single subject (all its trials):

.. code-block:: bash

    python scripts/train.py \\
        --data-dir data/raw \\
        --subjects AB01 \\
        --model-type xgb

Multiple specific subjects:

.. code-block:: bash

    python scripts/train.py \\
        --data-dir data/raw \\
        --subjects AB01 AB02 AB03 \\
        --model-type lr \\
        --window-size 200 \\
        --step-size 100 \\
        --features cadence stride_symmetry spectral

All subjects:

.. code-block:: bash

    python scripts/train.py \\
        --data-dir data/raw \\
        --subjects all \\
        --model-type svm
"""

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from gait_dynamics.data.loader import list_subjects, load_subject
from gait_dynamics.data.preprocess import bandpass_filter, window_signal
from gait_dynamics.features.extraction import FEATURE_REGISTRY, build_feature_matrix
from gait_dynamics.models.sklearn_pipeline import build_pipeline, save_pipeline, train_evaluate

# Paths are resolved relative to the project root so the script is callable
# from any working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DB_PATH = _PROJECT_ROOT / "db" / "runs.db"
_MODELS_DIR = _PROJECT_ROOT / "models"


# ── Private helpers ───────────────────────────────────────────────────────────

def _list_trials(data_dir: str | Path, subject_id: str) -> list[str]:
    """
    Return a sorted list of trial identifiers for a single subject.

    A trial is identified by the stem of any ``.hea`` WFDB header file found
    directly inside the subject's directory.

    Parameters
    ----------
    data_dir : str or Path
        Root directory of the dataset.
    subject_id : str
        Subject identifier matching a subdirectory name (e.g. ``"AB01"``).

    Returns
    -------
    list[str]
        Sorted list of trial stems (e.g. ``["AB01_LG_R01", "AB01_SD_R01"]``).
    """
    subject_dir = Path(data_dir) / subject_id
    return sorted(f.stem for f in subject_dir.glob("*.hea"))


def _subjects_key(subjects_arg: list[str], resolved: list[str]) -> str:
    """
    Derive a stable filename stem from the resolved subject list.

    Parameters
    ----------
    subjects_arg : list[str]
        Raw ``--subjects`` value as passed by the user.
    resolved : list[str]
        Actual subject IDs after expanding ``"all"``.

    Returns
    -------
    str
        ``"all"`` when ``subjects_arg`` was ``["all"]``, the single subject ID
        when exactly one subject is resolved, or
        ``"multi_N_subjects"`` for two or more.
    """
    if subjects_arg == ["all"]:
        return "all"
    if len(resolved) == 1:
        return resolved[0]
    return f"multi_{len(resolved)}_subjects"


# ── SQLite helpers ────────────────────────────────────────────────────────────

def _init_db(db_path: Path) -> None:
    """
    Create the runs log table if it does not already exist.

    Parameters
    ----------
    db_path : Path
        Filesystem path to the SQLite database file.  Parent directories are
        created automatically.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                subject_id  TEXT    NOT NULL,
                trial_id    TEXT,
                model_type  TEXT    NOT NULL,
                features    TEXT    NOT NULL,
                window_size INTEGER NOT NULL,
                step_size   INTEGER NOT NULL,
                accuracy    REAL    NOT NULL,
                n_train     INTEGER NOT NULL,
                n_test      INTEGER NOT NULL,
                model_path  TEXT    NOT NULL
            )
            """
        )
        conn.commit()


def _log_run(
    db_path: Path,
    *,
    timestamp: str,
    subject_id: str,
    trial_id: str | None,
    model_type: str,
    features: list[str],
    window_size: int,
    step_size: int,
    accuracy: float,
    n_train: int,
    n_test: int,
    model_path: str,
) -> None:
    """
    Insert a single training run record into the SQLite log.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file (must already be initialised via
        :func:`_init_db`).
    timestamp : str
        ISO 8601 timestamp string for the run.
    subject_id : str
        Comma-separated list of all subject IDs included in the run
        (e.g. ``"AB01,AB02,AB03"`` or ``"all"``).
    trial_id : str or None
        Comma-separated ``"subject:trial"`` pairs for every loaded trial
        (e.g. ``"AB01:AB01_LG_R01,AB01:AB01_SD_R01"``).  ``None`` only if no
        trials were found.
    model_type : str
        Classifier identifier (``"xgb"``, ``"svm"``, or ``"lr"``).
    features : list[str]
        Feature names that were computed; stored as a comma-separated string.
    window_size : int
        Number of samples per window.
    step_size : int
        Number of samples advanced between windows.
    accuracy : float
        Test-set accuracy returned by :func:`train_evaluate`.
    n_train : int
        Number of training samples.
    n_test : int
        Number of test samples.
    model_path : str
        Absolute path to the saved ``.joblib`` file.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO runs
                (timestamp, subject_id, trial_id, model_type, features,
                 window_size, step_size, accuracy, n_train, n_test, model_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                subject_id,
                trial_id,
                model_type,
                ",".join(features),
                window_size,
                step_size,
                accuracy,
                n_train,
                n_test,
                model_path,
            ),
        )
        conn.commit()


# ── Pipeline orchestration ────────────────────────────────────────────────────

def run_training(args: argparse.Namespace) -> dict:
    """
    Execute the full gait classification training pipeline.

    Resolves subjects from ``args.subjects`` (expanding ``"all"`` via
    :func:`~gait_dynamics.data.loader.list_subjects`), loads every trial for
    each subject, preprocesses and extracts features, concatenates the results
    across all subject/trial pairs, trains and evaluates a classifier, then
    saves the model and logs the run.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.  Expected attributes:

        * ``data_dir`` (str)
        * ``subjects`` (list[str]) — one or more subject IDs, or ``["all"]``
        * ``model_type`` (str)
        * ``window_size`` (int)
        * ``step_size`` (int)
        * ``features`` (list[str])

    Returns
    -------
    dict
        Evaluation results as returned by :func:`train_evaluate`, augmented
        with the key ``model_path`` (str) pointing to the saved model file.
    """
    # ── 1. Resolve subject list ───────────────────────────────────────────────
    print("[1/5] Resolving subjects …")
    if args.subjects == ["all"]:
        subject_ids = list_subjects(args.data_dir)
        print(f"      'all' expanded → {len(subject_ids)} subjects found.")
    else:
        subject_ids = args.subjects
    print(f"      Subjects: {subject_ids}")

    # ── 2. Load, preprocess, and extract features per subject/trial ───────────
    print(
        f"[2/5] Loading all trials, filtering, windowing "
        f"(window={args.window_size}, step={args.step_size}), "
        f"extracting features: {args.features} …"
    )
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    loaded_pairs: list[tuple[str, str]] = []  # (subject_id, trial_id)

    for subject_id in subject_ids:
        trial_ids = _list_trials(args.data_dir, subject_id)
        if not trial_ids:
            print(f"      {subject_id}: no trials found — skipping.")
            continue

        for trial_id in trial_ids:
            subject_data = load_subject(
                args.data_dir,
                subject_id,
                trial_id=trial_id,
            )
            acc: np.ndarray = subject_data["acc"]
            sample_rate: int = subject_data["sample_rate"]

            acc_filtered = bandpass_filter(acc, sample_rate)
            windows = window_signal(acc_filtered, args.window_size, args.step_size)
            n_windows = windows.shape[0]

            X_trial = build_feature_matrix(windows, sample_rate, args.features)

            # TODO: Replace with real per-window activity labels loaded from
            #       the Camargo et al. dataset annotation files (e.g. the
            #       task/condition columns in the accompanying CSV metadata).
            #       The placeholder below assigns every window to class 0
            #       (level walking).  Using all-zero labels will cause
            #       train_evaluate to raise a ValueError from sklearn's
            #       StratifiedShuffleSplit because only one class is present.
            y_trial = np.zeros(n_windows, dtype=int)

            X_parts.append(X_trial)
            y_parts.append(y_trial)
            loaded_pairs.append((subject_id, trial_id))
            print(f"      {subject_id} / {trial_id}: {n_windows} windows.")

    if not X_parts:
        raise RuntimeError("No windows were produced. Check data_dir and subject list.")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    print(f"      Combined feature matrix: {X.shape}  labels: {y.shape}")

    # ── 3. Train & evaluate ───────────────────────────────────────────────────
    print(f"[3/5] Training '{args.model_type}' pipeline …")
    pipeline = build_pipeline(args.model_type)
    results = train_evaluate(pipeline, X, y)
    print(
        f"      Accuracy: {results['accuracy']:.4f}  "
        f"(train={results['n_train']}, test={results['n_test']})"
    )

    # ── 4. Save model ─────────────────────────────────────────────────────────
    subjects_key = _subjects_key(args.subjects, subject_ids)
    model_path = _MODELS_DIR / f"{subjects_key}_{args.model_type}.joblib"
    print(f"[4/5] Saving pipeline to '{model_path}' …")
    save_pipeline(pipeline, model_path)

    # ── 5. Log run to SQLite ──────────────────────────────────────────────────
    _init_db(_DB_PATH)
    timestamp = datetime.now().isoformat()
    _log_run(
        _DB_PATH,
        timestamp=timestamp,
        subject_id=",".join(subject_ids),
        trial_id=",".join(f"{s}:{t}" for s, t in loaded_pairs) or None,
        model_type=args.model_type,
        features=args.features,
        window_size=args.window_size,
        step_size=args.step_size,
        accuracy=results["accuracy"],
        n_train=results["n_train"],
        n_test=results["n_test"],
        model_path=str(model_path.resolve()),
    )
    print(f"[5/5] Run logged to '{_DB_PATH}'.")

    results["model_path"] = str(model_path.resolve())
    return results


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and run the training pipeline."""
    parser = argparse.ArgumentParser(
        prog="train",
        description=(
            "Train a gait activity classifier from IMU data. "
            "Pass --subjects all to train on every subject in --data-dir, "
            "or list specific subject IDs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        metavar="PATH",
        help="Root directory of the Camargo et al. PhysioNet dataset.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        required=True,
        metavar="ID",
        help=(
            "One or more subject IDs (e.g. AB01 AB02), or the special value "
            "'all' to include every subject found in --data-dir."
        ),
    )
    parser.add_argument(
        "--model-type",
        default="xgb",
        choices=["xgb", "svm", "lr"],
        help="Classifier type.",
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
    results = run_training(args)

    print("\n── Results ──────────────────────────────────────")
    print(results["classification_report"])


if __name__ == "__main__":
    main()
