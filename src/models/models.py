"""
models.py
-------------
ModelFactory: returns unfitted scikit-learn Pipelines for each supported model.

Design rules enforced here:
  - StandardScaler is always the first step inside the pipeline.
    It must NEVER be fitted outside the pipeline — doing so would allow the
    scaler to see validation-fold data during GridSearchCV, causing leakage.
  - Every pipeline is returned unfitted so GridSearchCV handles all fitting.
  - The estimator step is named "classifier" so hyperparameter grids in
    config.py can use the "classifier__<param>" prefix convention uniformly.
  - XGBoost verbosity is set to 0 to suppress per-tree training logs.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

from config import CV_CONFIG

SUPPORTED_MODELS = ["logistic_regression", "svc", "knn", "xgboost"]


def build_pipeline(model_name: str) -> Pipeline:
    """
    Return an unfitted scikit-learn Pipeline for the given model name.

    The pipeline always has two steps:
      1. "scaler"     -> StandardScaler (fitted only inside CV folds)
      2. "classifier" -> the requested estimator

    Args:
        model_name: One of "logistic_regression", "svc", "knn", "xgboost".

    Returns:
        An unfitted sklearn Pipeline ready to be passed to GridSearchCV.

    Raises:
        ValueError: If model_name is not in SUPPORTED_MODELS.
        ImportError: If "xgboost" is requested but xgboost is not installed.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Supported models: {SUPPORTED_MODELS}"
        )

    seed = CV_CONFIG["random_seed"]

    if model_name == "logistic_regression":
        estimator = LogisticRegression(
            random_state=seed,
            max_iter=1000,
        )

    elif model_name == "svc":
        # probability=True enables predict_proba if needed for calibration later
        estimator = SVC(
            random_state=seed,
            probability=True,
        )

    elif model_name == "knn":
        # Default n_neighbors=5; GridSearchCV will override this via the
        # hyperparameter grid. The KNN neighbor cap check is enforced in
        # evaluation.py before the search begins.
        estimator = KNeighborsClassifier(
            n_neighbors=5,
        )

    elif model_name == "xgboost":
        if not _XGBOOST_AVAILABLE:
            raise ImportError(
                "xgboost is not installed. Run: pip install xgboost"
            )
        estimator = XGBClassifier(
            random_state=seed,
            verbosity=0,          # suppress per-tree training output
            eval_metric="mlogloss",
            use_label_encoder=False,
        )

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )