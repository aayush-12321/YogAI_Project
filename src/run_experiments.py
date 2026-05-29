"""
run_experiments.py
------------------
Central driver for the YogAI comparative ML study.

Iterates over every (pose, model) combination, runs a full hyperparameter
search experiment, serializes all artifacts immediately after each run,
and produces a consolidated results CSV and benchmark table.

Artifacts saved per experiment (all in the current working directory):
  {pose}_{model}_pipeline.joblib     — fitted sklearn Pipeline (scaler + estimator)
  {pose}_{model}_label_encoder.joblib — fitted LabelEncoder for decoding predictions
  {pose}_{model}_class_map.json      — human-readable int->label mapping
  {pose}_{model}_metadata.json       — feature list, best params, CV score, latency

  results_experiment_summary.csv     — aggregated benchmark table (all runs)

Usage:
  python run_experiments.py --csv path/to/master_dataset.csv
  python run_experiments.py --csv path/to/master_dataset.csv --poses plank warrior_ii
  python run_experiments.py --csv path/to/master_dataset.csv --models logistic_regression svc
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

# Allow imports from src/ without installing as a package
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import POSE_FEATURES, CV_CONFIG
from src.datasets.dataset_loader import load_pose_data
from src.evaluation.evaluation import run_experiment
from src.models.models import SUPPORTED_MODELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Columns to include in the printed benchmark table
BENCHMARK_COLUMNS = [
    "pose",
    "model",
    "best_cv_score",
    "test_f1_weighted",
    "test_precision_weighted",
    "test_recall_weighted",
    "test_accuracy",
    "mean_ms",
    "p95_ms",
    "p99_ms",
    "implied_max_fps",
    "n_train",
    "n_test",
]


def save_artifacts(
    result: dict,
    label_encoder,
    class_map: dict[int, str],
    feature_columns: list[str],
) -> None:
    """
    Serialize all model artifacts for a completed experiment to disk.

    Artifacts are saved immediately after each experiment (not batched at the
    end) so that a crash mid-run does not lose completed work.

    Saved files:
      {pose}_{model}_pipeline.joblib      — reusable inference pipeline
      {pose}_{model}_label_encoder.joblib — for decoding integer predictions
      {pose}_{model}_class_map.json       — human-readable label mapping
      {pose}_{model}_metadata.json        — all reproducibility metadata

    Args:
        result:          The dict returned by run_experiment().
        label_encoder:   Fitted LabelEncoder from load_pose_data().
        class_map:       Int->string label mapping from load_pose_data().
        feature_columns: The ordered list of feature columns used for this pose.
    """
    pose = result["pose"]
    model = result["model"]
    prefix = f"{pose}_{model}"

    pipeline = result["_best_pipeline"]

    joblib.dump(pipeline, f"{prefix}_pipeline.joblib")
    logger.info("Saved pipeline -> %s_pipeline.joblib", prefix)

    joblib.dump(label_encoder, f"{prefix}_label_encoder.joblib")
    logger.info("Saved label encoder -> %s_label_encoder.joblib", prefix)

    # class_map keys are ints; JSON requires string keys
    with open(f"{prefix}_class_map.json", "w") as f:
        json.dump({str(k): v for k, v in class_map.items()}, f, indent=2)
    logger.info("Saved class map -> %s_class_map.json", prefix)

    metadata = {
        "pose": pose,
        "model": model,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "best_params": result["best_params"],
        "best_cv_score": result["best_cv_score"],
        "test_f1_weighted": result["test_f1_weighted"],
        "test_accuracy": result["test_accuracy"],
        "n_train": result["n_train"],
        "n_test": result["n_test"],
        "cv_config": CV_CONFIG,
        "latency": {
            "mean_ms": result["mean_ms"],
            "std_ms": result["std_ms"],
            "p95_ms": result["p95_ms"],
            "p99_ms": result["p99_ms"],
            "implied_max_fps": result["implied_max_fps"],
        },
        "class_map": {str(k): v for k, v in class_map.items()},
        "classification_report": result["classification_report"],
        "confusion_matrix": result["confusion_matrix"],
    }

    with open(f"{prefix}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata -> %s_metadata.json", prefix)


def print_benchmark_table(df: pd.DataFrame) -> None:
    """
    Print a formatted side-by-side benchmark table to stdout.

    Args:
        df: DataFrame containing all experiment results.
    """
    available = [c for c in BENCHMARK_COLUMNS if c in df.columns]
    display = df[available].copy()

    # Format floats for readability
    float_cols = [
        "best_cv_score", "test_f1_weighted", "test_precision_weighted",
        "test_recall_weighted", "test_accuracy",
        "mean_ms", "p95_ms", "p99_ms", "implied_max_fps",
    ]
    for col in float_cols:
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{x:.4f}" if isinstance(x, float) else x)

    separator = "-" * 120
    print("\n" + separator)
    print("YOGAI — EXPERIMENT BENCHMARK RESULTS")
    print(separator)
    print(display.to_string(index=False))
    print(separator + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YogAI ML experiments.")
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the master CSV dataset.",
    )
    parser.add_argument(
        "--poses",
        nargs="+",
        default=list(POSE_FEATURES.keys()),
        help="Poses to run experiments for. Defaults to all poses in config.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=SUPPORTED_MODELS,
        help="Models to evaluate. Defaults to all supported models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)

    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        sys.exit(1)

    all_results: list[dict] = []
    total = len(args.poses) * len(args.models)
    completed = 0

    for pose_name in args.poses:
        # Load pose data once per pose; reuse across all models
        logger.info("Loading data for pose: %s", pose_name)
        try:
            X, y, groups, label_encoder, class_map = load_pose_data(pose_name, csv_path)
        except (ValueError, KeyError) as exc:
            logger.error("Failed to load pose '%s': %s — skipping.", pose_name, exc)
            continue

        feature_columns = list(POSE_FEATURES[pose_name])

        for model_name in args.models:
            completed += 1
            logger.info(
                "[%d/%d] Running experiment: pose=%s, model=%s",
                completed, total, pose_name, model_name,
            )

            try:
                result = run_experiment(
                    pose_name=pose_name,
                    model_name=model_name,
                    X=X,
                    y=y,
                    groups=groups,
                )
            except Exception as exc:
                logger.error(
                    "Experiment failed [%s | %s]: %s — skipping.",
                    pose_name, model_name, exc,
                )
                continue

            # Serialize artifacts immediately — do not wait until the loop ends
            save_artifacts(result, label_encoder, class_map, feature_columns)

            # Strip the private pipeline key before storing in results list
            public_result = {k: v for k, v in result.items() if not k.startswith("_")}
            all_results.append(public_result)

    if not all_results:
        logger.error("No experiments completed successfully. Check errors above.")
        sys.exit(1)

    # Aggregate and export results
    results_df = pd.DataFrame(all_results)

    # Drop nested columns not suited for flat CSV
    csv_df = results_df.drop(
        columns=["best_params", "classification_report", "classification_report_dict", "confusion_matrix"],
        errors="ignore",
    )
    output_csv = "results_experiment_summary.csv"
    csv_df.to_csv(output_csv, index=False)
    logger.info("Results saved -> %s", output_csv)

    print_benchmark_table(csv_df)

    logger.info(
        "All experiments complete. %d/%d combinations succeeded.",
        len(all_results),
        total,
    )


if __name__ == "__main__":
    main()