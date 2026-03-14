"""
Scikit-learn ML pipeline for 5-class gait activity classification.

Target activity classes
-----------------------
0 — Level walking
1 — Stair ascent
2 — Stair descent
3 — Ramp ascent
4 — Ramp descent

Supported classifiers are selected by the ``model_type`` key and sit behind
a ``StandardScaler`` in a single ``sklearn.pipeline.Pipeline`` so that
scaling coefficients are always fitted on training data only.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


_VALID_MODEL_TYPES = ("xgb", "svm", "lr")

_ACTIVITY_LABELS = [
    "level_walking",
    "stair_ascent",
    "stair_descent",
    "ramp_ascent",
    "ramp_descent",
]


def build_pipeline(model_type: str = "xgb") -> Pipeline:
    """
    Build a scikit-learn Pipeline with StandardScaler and a classifier.

    The pipeline step names are ``"scaler"`` and ``model_type``, making
    the chosen classifier identifiable from the pipeline object alone.

    Parameters
    ----------
    model_type : str, optional
        Classifier to use. One of:

        ``"xgb"``
            XGBClassifier with ``n_estimators=100``, ``random_state=42``,
            and ``eval_metric="mlogloss"``.
        ``"svm"``
            SVC with ``kernel="rbf"``, ``random_state=42``,
            and ``probability=True``.
        ``"lr"``
            LogisticRegression with ``max_iter=1000``, ``random_state=42``

        Default is ``"xgb"``.

    Returns
    -------
    Pipeline
        Unfitted pipeline ready for ``fit`` / ``predict``.

    Raises
    ------
    ValueError
        If ``model_type`` is not one of the supported values.
    """
    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Valid options are: {list(_VALID_MODEL_TYPES)}"
        )

    if model_type == "xgb":
        classifier = XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0,
        )
    elif model_type == "svm":
        classifier = SVC(
            kernel="rbf",
            random_state=42,
            probability=True,
    )
    else:  # "lr"
        classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
        )

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (model_type, classifier),
        ]
    )


def train_evaluate(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Split data, fit the pipeline on the training set, and evaluate on the test set.

    Parameters
    ----------
    pipeline : Pipeline
        An unfitted scikit-learn Pipeline, typically from :func:`build_pipeline`.
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    y : np.ndarray
        Integer class labels of shape ``(n_samples,)`` with values in
        ``{0, 1, 2, 3, 4}`` corresponding to the five activity classes.
    test_size : float, optional
        Fraction of samples reserved for evaluation. Default is ``0.2``.
    random_state : int, optional
        Random seed for reproducible train/test splitting. Default is ``42``.

    Returns
    -------
    dict
        Dictionary with the following keys:

        ``model_type`` : str
            Name of the last pipeline step (i.e. the classifier identifier).
        ``accuracy`` : float
            Classification accuracy on the held-out test set.
        ``classification_report`` : str
            Full per-class precision/recall/F1 report from
            ``sklearn.metrics.classification_report``.
        ``n_train`` : int
            Number of training samples.
        ``n_test`` : int
            Number of test samples.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # The last step name is the model_type string set in build_pipeline.
    model_type = pipeline.steps[-1][0]

    return {
        "model_type": model_type,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=_ACTIVITY_LABELS,
            zero_division=0,
        ),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def save_pipeline(pipeline: Pipeline, path: str | Path) -> None:
    """
    Serialize a fitted pipeline to disk using joblib.

    Parent directories are created automatically if they do not exist.

    Parameters
    ----------
    pipeline : Pipeline
        A fitted scikit-learn Pipeline to persist.
    path : str or Path
        Destination file path (e.g. ``"models/gait_xgb.joblib"``).

    Returns
    -------
    None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
