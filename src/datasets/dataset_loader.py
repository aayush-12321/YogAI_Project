# NOT IN USE

"""
src/datasets/dataset_loader.py
------------------
Responsible for loading and isolating pose-specific data from the master CSV.

Key guarantees:
  - Only rows belonging to the requested pose are returned.
  - Only that pose's designated feature columns are returned (no cross-contamination).
  - source_id is used as the group identifier to prevent video-level data leakage
    across train/test folds (GroupKFold). Frames from the same video share a
    source_id, so they are always kept together in the same fold.
  - Labels are encoded to integers using a fitted LabelEncoder, and both the
    encoder and the class mapping are returned so inference can replicate the
    exact same encoding without refitting.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import (
    FRAME_NUMBER_COLUMN,
    GROUP_COLUMNS,
    LABEL_COLUMN,
    POSE_FEATURES,
    SOURCE_ID_COLUMN,
)

logger = logging.getLogger(__name__)


def load_pose_data(
    pose_name: str,
    csv_path: str | Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None, LabelEncoder, dict[int, str]]:
    """
    Load and isolate data for a single yoga pose from the master CSV.

    The function performs the following steps:
      1. Validates the pose name against POSE_FEATURES.
      2. Loads the CSV and filters rows to those belonging to `pose_name`.
      3. Selects only the feature columns designated for that pose.
      4. Encodes string labels to consecutive integers using LabelEncoder.
      5. Detects whether a group column is present (source_id, session_id, or
         participant_id) and returns group labels for GroupKFold if so.

    Args:
        pose_name: One of the keys in POSE_FEATURES (e.g. "plank").
        csv_path:  Path to the master CSV containing all poses.

    Returns:
        X:             DataFrame of shape (n_samples, n_pose_features).
        y:             Integer-encoded label array of shape (n_samples,).
        groups:        Array of group IDs (source_id values) if a group column
                       is present, else None. Pass to GroupKFold.
        label_encoder: Fitted LabelEncoder. Save this with the model artifact
                       so inference can decode predictions back to strings.
        class_map:     Dict mapping encoded integer -> original string label
                       (e.g. {0: "correct", 1: "incorrect", 2: "partial"}).

    Raises:
        ValueError: If pose_name is not in POSE_FEATURES.
        ValueError: If required feature columns are missing from the CSV.
        ValueError: If no rows exist for the requested pose.
    """
    if pose_name not in POSE_FEATURES:
        raise ValueError(
            f"Unknown pose '{pose_name}'. "
            f"Valid poses: {list(POSE_FEATURES.keys())}"
        )

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Filter to rows belonging to this pose
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Column '{LABEL_COLUMN}' not found in {csv_path}.")

    # The pose name is assumed to be part of the label string, e.g. "plank_correct"
    # OR stored in a separate "pose" column. Support both conventions.
    if "pose" in df.columns:
        pose_df = df[df["pose"].str.lower() == pose_name.lower()].copy()
    else:
        pose_df = df[df[LABEL_COLUMN].str.lower().str.startswith(pose_name.lower())].copy()

    if pose_df.empty:
        raise ValueError(
            f"No rows found for pose '{pose_name}' in {csv_path}. "
            f"Check that either a 'pose' column exists or that label values "
            f"are prefixed with the pose name."
        )

    # Validate all expected feature columns are present
    expected_features = POSE_FEATURES[pose_name]
    missing = [col for col in expected_features if col not in pose_df.columns]
    if missing:
        raise ValueError(
            f"The following expected feature columns for pose '{pose_name}' "
            f"are missing from the CSV: {missing}"
        )

    X = pose_df[expected_features].copy()

    # Encode labels to integers
    # Strip any pose-name prefix from labels if present
    # e.g. "plank_correct" -> "correct"
    raw_labels = pose_df[LABEL_COLUMN].str.lower()
    if "pose" not in df.columns:
        prefix = pose_name.lower() + "_"
        raw_labels = raw_labels.str.removeprefix(prefix)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(raw_labels)
    class_map = {idx: cls for idx, cls in enumerate(label_encoder.classes_)}

    logger.info(
        "Pose '%s': %d samples, %d features, classes=%s",
        pose_name,
        len(X),
        len(expected_features),
        class_map,
    )

    # Detect group column for GroupKFold (prevents video-level leakage)
    groups = None
    detected_group_col = None
    for col in GROUP_COLUMNS:
        if col in pose_df.columns:
            detected_group_col = col
            break

    if detected_group_col:
        groups = pose_df[detected_group_col].values
        n_unique_groups = len(np.unique(groups))
        logger.info(
            "Group column '%s' detected. %d unique groups found. "
            "GroupKFold will be used to prevent video-level leakage.",
            detected_group_col,
            n_unique_groups,
        )
    else:
        warnings.warn(
            f"No group column found for pose '{pose_name}'. "
            f"Checked for: {GROUP_COLUMNS}. "
            f"StratifiedKFold will be used. If your data contains video frames, "
            f"add a '{SOURCE_ID_COLUMN}' column to prevent temporal leakage.",
            UserWarning,
            stacklevel=2,
        )

    return X, y, groups, label_encoder, class_map