"""
evaluation.py
-----------------
Core experiment engine for the YogAI comparative study.

Responsibilities:
  - Run GridSearchCV (or RandomizedSearchCV for XGBoost) over unfitted pipelines.
  - Select GroupKFold or StratifiedKFold based on whether group labels are present.
  - Enforce KNN neighbor cap with a runtime warning before search begins.
  - Measure per-sample inference latency with warm-up calls excluded.
  - Return a fully populated results dictionary per experiment.
"""

from __future__ import annotations

import logging
import math
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)

from src.config import (
    CV_CONFIG,
    HYPERPARAMETER_GRIDS,
    XGBOOST_RANDOM_SEARCH_ITER,
)
from src.models import build_pipeline

logger = logging.getLogger(__name__)

LATENCY_WARMUP_CALLS = 10
LATENCY_TRIAL_CALLS = 500


def _check_knn_neighbor_cap(n_neighbors_grid: list[int], n_training_samples: int) -> None:
    """
    Warn if any KNN n_neighbors value in the grid exceeds a safe ceiling.

    The ceiling is floor(sqrt(n_training_samples)), a standard heuristic.
    Values above this are not invalid but are unlikely to generalize and
    produce meaninglessly large neighborhood lookups.

    Args:
        n_neighbors_grid:   List of n_neighbors values from the hyperparameter grid.
        n_training_samples: Number of samples in the training set.
    """
    ceiling = math.floor(math.sqrt(n_training_samples))
    oversized = [k for k in n_neighbors_grid if k > ceiling]
    if oversized:
        warnings.warn(
            f"KNN grid contains n_neighbors values {oversized} that exceed "
            f"the recommended ceiling of floor(sqrt({n_training_samples})) = {ceiling}. "
            f"These values are legal but unlikely to improve generalization. "
            f"Consider trimming the grid.",
            UserWarning,
            stacklevel=3,
        )


def _measure_inference_latency(
    pipeline,
    X_probe: np.ndarray,
    n_warmup: int = LATENCY_WARMUP_CALLS,
    n_trials: int = LATENCY_TRIAL_CALLS,
) -> dict[str, float]:
    """
    Measure per-sample prediction latency of a fitted pipeline.

    Warm-up calls are discarded to eliminate JIT compilation and CPU cache
    cold-start effects that would inflate the first few measurements.
    p95 and p99 are reported alongside the mean because real-time systems
    are bounded by tail latency, not average latency. A model with a low
    mean but high p99 will produce visible stuttering on a 30fps webcam feed.

    Args:
        pipeline:  A fitted sklearn Pipeline.
        X_probe:   A single sample as a 2D array of shape (1, n_features).
        n_warmup:  Number of discarded warm-up predictions.
        n_trials:  Number of timed prediction calls.

    Returns:
        Dict with keys: mean_ms, std_ms, p95_ms, p99_ms, implied_max_fps.
    """
    for _ in range(n_warmup):
        pipeline.predict(X_probe)

    latencies_ms = np.empty(n_trials)
    for i in range(n_trials):
        t0 = time.perf_counter()
        pipeline.predict(X_probe)
        latencies_ms[i] = (time.perf_counter() - t0) * 1000.0

    return {
        "mean_ms": round(float(np.mean(latencies_ms)), 4),
        "std_ms": round(float(np.std(latencies_ms)), 4),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 4),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 4),
        "implied_max_fps": round(1000.0 / float(np.mean(latencies_ms)), 1),
    }


def run_experiment(
    pose_name: str,
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None = None,
) -> dict:
    """
    Run a full hyperparameter search experiment for one pose/model combination.

    Steps:
      1. Split data into train and held-out test sets (stratified, group-aware).
      2. Build an unfitted pipeline via ModelFactory.
      3. Enforce the KNN neighbor cap check if applicable.
      4. Run GridSearchCV (or RandomizedSearchCV for XGBoost) on the train set.
      5. Evaluate best_estimator_ on the held-out test set.
      6. Measure inference latency on a single test sample.

    The held-out test set uses 20% of samples. When groups are present,
    GroupShuffleSplit semantics are respected so no source_id appears in both
    train and test.

    Args:
        pose_name:  Pose name string (for logging and result labeling).
        model_name: One of the supported model names.
        X:          Feature DataFrame for this pose.
        y:          Integer-encoded label array.
        groups:     Optional source_id group array for GroupKFold.

    Returns:
        A dict containing:
          pose, model, best_params, best_cv_score, classification_report (str),
          classification_report_dict, confusion_matrix, class_labels,
          n_train, n_test, and all 5 latency metrics.
    """
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    seed = CV_CONFIG["random_seed"]
    n_splits = CV_CONFIG["n_splits"]

    # Train/test split — keep video frames together when groups are present
    if groups is not None:
        unique_groups = np.unique(groups)
        n_test_groups = max(1, int(len(unique_groups) * 0.2))

        rng = np.random.default_rng(seed)
        test_groups = set(rng.choice(unique_groups, size=n_test_groups, replace=False))

        test_mask = np.isin(groups, list(test_groups))
        train_mask = ~test_mask

        X_train, X_test = X_arr[train_mask], X_arr[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        groups_train = groups[train_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y, test_size=0.2, stratify=y, random_state=seed
        )
        groups_train = None

    n_train = len(X_train)
    n_test = len(X_test)

    logger.info(
        "[%s | %s] Train: %d samples, Test: %d samples",
        pose_name,
        model_name,
        n_train,
        n_test,
    )

    # Build unfitted pipeline — must not be pre-fitted before GridSearchCV
    pipeline = build_pipeline(model_name)

    # KNN neighbor cap enforcement
    if model_name == "knn":
        knn_grid = HYPERPARAMETER_GRIDS["knn"]["classifier__n_neighbors"]
        _check_knn_neighbor_cap(knn_grid, n_train)

    # Select CV strategy
    if groups_train is not None:
        cv = GroupKFold(n_splits=min(n_splits, len(np.unique(groups_train))))
        cv_kwargs = {"groups": groups_train}
        logger.info("[%s | %s] Using GroupKFold (video-safe splits)", pose_name, model_name)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        cv_kwargs = {}
        logger.info("[%s | %s] Using StratifiedKFold", pose_name, model_name)

    param_grid = HYPERPARAMETER_GRIDS[model_name]

    # XGBoost uses RandomizedSearchCV due to the larger search space
    if model_name == "xgboost":
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=XGBOOST_RANDOM_SEARCH_ITER,
            scoring=CV_CONFIG["scoring"],
            cv=cv,
            n_jobs=CV_CONFIG["n_jobs"],
            random_state=seed,
            refit=True,
            verbose=0,
        )
    else:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=CV_CONFIG["scoring"],
            cv=cv,
            n_jobs=CV_CONFIG["n_jobs"],
            refit=True,
            verbose=0,
        )

    search.fit(X_train, y_train, **cv_kwargs)

    best_pipeline = search.best_estimator_

    # Evaluate on held-out test set
    y_pred = best_pipeline.predict(X_test)

    clf_report_str = classification_report(y_test, y_pred, zero_division=0)
    clf_report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    logger.info(
        "[%s | %s] Best CV score: %.4f | Test F1 (weighted): %.4f",
        pose_name,
        model_name,
        search.best_score_,
        clf_report_dict["weighted avg"]["f1-score"],
    )

    # Inference latency on a single sample
    X_probe = X_test[0:1]
    latency_metrics = _measure_inference_latency(best_pipeline, X_probe)

    logger.info(
        "[%s | %s] Latency — mean: %.3f ms | p99: %.3f ms | implied FPS: %.1f",
        pose_name,
        model_name,
        latency_metrics["mean_ms"],
        latency_metrics["p99_ms"],
        latency_metrics["implied_max_fps"],
    )

    return {
        "pose": pose_name,
        "model": model_name,
        "best_params": search.best_params_,
        "best_cv_score": round(search.best_score_, 4),
        "test_f1_weighted": round(clf_report_dict["weighted avg"]["f1-score"], 4),
        "test_precision_weighted": round(clf_report_dict["weighted avg"]["precision"], 4),
        "test_recall_weighted": round(clf_report_dict["weighted avg"]["recall"], 4),
        "test_accuracy": round(clf_report_dict["accuracy"], 4),
        "classification_report": clf_report_str,
        "classification_report_dict": clf_report_dict,
        "confusion_matrix": conf_matrix,
        "n_train": n_train,
        "n_test": n_test,
        **latency_metrics,
        # best_pipeline is returned so run_experiments.py can serialize it
        "_best_pipeline": best_pipeline,
    }