"""
Feature extraction for the MASTER pose-type classifier (mountain vs plank vs warrior2).

This is a different problem from the per-pose error classifiers: here the model only
needs to answer "which pose is this?", so the target label is the POSE NAME, not the
per-pose error class (correct / high_back / low_back / etc). The per-pose error label
in each raw CSV is preserved as `original_label` for debugging/auditing, but it is
never used as the training target for this model.

Two raw CSV schemas are supported and normalised into one canonical layout:

  1. Indexed schema  (mountain, warrior2, ...):
     source_id, label, frame_number, x_0, y_0, z_0, v_0, ..., x_32, y_32, z_32, v_32

  2. Named schema     (plank):
     label, nose_x, nose_y, nose_z, nose_v, left_shoulder_x, ..., right_foot_index_v
     (no source_id / frame_number columns)

Both schemas cover the same 17 landmarks needed for the master feature set
(nose, shoulders, elbows, wrists, hips, knees, ankles, heels, foot indices), so once
the named schema is remapped to indexed column names (`nose_x` -> `x_0`, etc.) the two
are feature-identical.

Unlike the warrior2 script, no front/back orientation resolution is needed here.
Pose TYPE does not depend on which leg is forward, so every feature is computed
directly from fixed left/right landmark indices.
"""

import os

import numpy as np
import pandas as pd
import yaml

YAML_FILE = "../../configs/poses/master_model.yaml"
OUTPUT_DIR = None

# Raw CSV sources: {pose_name: path_to_raw_csv}. pose_name becomes the training label.
RAW_SOURCES = {
    # "mountain": "../../data/annotations/mountain_pose/mountain_pose_raw.csv",
    # "plank": "../../data/annotations/plank_pose/plank_train.csv",
    # "warrior2": "../../data/annotations/warrior2_pose/warrior2_pose_raw.csv",
    "plank": "../../data/annotations/plank_pose/plank_test.csv",

}

METADATA_COLS = ["source_id", "label", "original_label", "frame_number"]
EPSILON = 1e-6

# MediaPipe landmark name -> index, for the subset the plank CSV exports.
_PLANK_LANDMARK_INDEX = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


def _standardize_named_schema(df: pd.DataFrame, pose_name: str) -> pd.DataFrame:
    """
    Convert the plank-style named-column CSV (nose_x, left_shoulder_y, ...) into the
    canonical indexed layout (x_0, y_11, ...) plus synthesised source_id/frame_number.

    Args:
        df:        Raw plank dataframe with named landmark columns and a `label` column.
        pose_name: Constant pose-type label to assign to every row ("plank").

    Returns:
        Canonical dataframe: source_id, label, original_label, frame_number, x_i/y_i/z_i/v_i.
    """
    rename_map: dict[str, str] = {}
    for landmark_name, idx in _PLANK_LANDMARK_INDEX.items():
        for axis in ("x", "y", "z", "v"):
            src_col = f"{landmark_name}_{axis}"
            if src_col not in df.columns:
                raise ValueError(
                    f"Plank CSV is missing expected column '{src_col}' for landmark "
                    f"'{landmark_name}' (index {idx})."
                )
            rename_map[src_col] = f"{axis}_{idx}"

    landmark_cols = list(rename_map.keys())
    out = df[landmark_cols].rename(columns=rename_map).copy()

    out.insert(0, "frame_number", np.arange(len(df)))
    out.insert(0, "original_label", df["label"].to_numpy())
    out.insert(0, "label", pose_name)
    out.insert(0, "source_id", pose_name)
    return out


def _standardize_indexed_schema(df: pd.DataFrame, pose_name: str) -> pd.DataFrame:
    """
    Convert an indexed-schema CSV (mountain, warrior2) into the canonical layout.

    The per-pose error label is renamed to `original_label` and replaced with the
    constant pose-type label used to train the master classifier.

    Args:
        df:        Raw dataframe with source_id, label, frame_number, x_i/y_i/z_i/v_i.
        pose_name: Constant pose-type label to assign to every row.

    Returns:
        Canonical dataframe matching _standardize_named_schema's output columns.
    """
    landmark_cols = [c for c in df.columns if c not in ("source_id", "label", "frame_number")]
    out = df[landmark_cols].copy()

    out.insert(0, "frame_number", df["frame_number"].to_numpy())
    out.insert(0, "original_label", df["label"].to_numpy())
    out.insert(0, "label", pose_name)
    out.insert(
        0,
        "source_id",
        df["source_id"].to_numpy() if "source_id" in df.columns else pose_name,
    )
    return out


def load_and_standardize(raw_csv_path: str, pose_name: str) -> pd.DataFrame:
    """
    Load a raw landmark CSV of either supported schema and normalise it to the
    canonical layout used by the master feature pipeline.

    Schema is detected by column presence: if `source_id` and an indexed column like
    `x_0` are both present, it's the indexed schema; otherwise it's treated as the
    named schema (plank).

    Args:
        raw_csv_path: Path to the raw CSV.
        pose_name:    Pose-type label to assign ("mountain", "plank", "warrior2").

    Returns:
        Canonical dataframe: source_id, label, original_label, frame_number, landmark columns.
    """
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw landmarks CSV not found: '{raw_csv_path}'")

    df = pd.read_csv(raw_csv_path)

    is_indexed_schema = "source_id" in df.columns and "x_0" in df.columns
    if is_indexed_schema:
        standardized = _standardize_indexed_schema(df, pose_name)
    else:
        if "label" not in df.columns:
            raise ValueError(
                f"'{raw_csv_path}' matches neither known schema: missing 'label' column."
            )
        standardized = _standardize_named_schema(df, pose_name)

    print(
        f"Loaded '{raw_csv_path}' as pose '{pose_name}' "
        f"({'indexed' if is_indexed_schema else 'named'} schema) -- {len(standardized)} rows."
    )
    return standardized


def build_master_raw_dataset(sources: dict[str, str]) -> pd.DataFrame:
    """
    Load and concatenate every pose's raw CSV into one canonical dataframe with
    a unified `label` column set to the pose name.

    Args:
        sources: {pose_name: raw_csv_path}.

    Returns:
        Concatenated canonical dataframe covering all poses.
    """
    frames = [load_and_standardize(path, pose_name) for pose_name, path in sources.items()]
    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined raw dataset: {len(combined)} rows across {len(sources)} poses.")
    return combined


def _validate_landmark_columns(df: pd.DataFrame, required_indices: set[int]) -> None:
    missing = [
        f"{axis}_{i}"
        for i in required_indices
        for axis in ("x", "y", "z")
        if f"{axis}_{i}" not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Standardized dataset is missing {len(missing)} expected landmark column(s): {missing}"
        )


def _collect_required_indices(feat_config: dict) -> set[int]:
    """Return every raw landmark index referenced anywhere in the feature config."""
    indices: set[int] = set()
    for section in ("joint_angles", "spatial_distances", "alignment_offsets"):
        for cfg in feat_config.get(section, []):
            for joint_list_key in ("joints", "normalization_factor"):
                for j in cfg.get(joint_list_key, []):
                    indices.add(int(j))
    return indices


def _xy(df: pd.DataFrame, idx: int) -> np.ndarray:
    return df[[f"x_{idx}", f"y_{idx}"]].to_numpy(dtype=np.float64)


def _col(df: pd.DataFrame, axis: str, idx: int) -> np.ndarray:
    return df[f"{axis}_{idx}"].to_numpy(dtype=np.float64)


def _angle_at_vertex_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """2D angle at vertex B, in degrees, for row-wise point triples."""
    ba = a - b
    bc = c - b
    dot = np.einsum("ij,ij->i", ba, bc)
    norm_ba = np.linalg.norm(ba, axis=1)
    norm_bc = np.linalg.norm(bc, axis=1)
    cosine = np.clip(dot / (norm_ba * norm_bc + EPSILON), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _euclidean_distance_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1)


def _point_to_line_distance_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    ap = p - a
    cross = np.abs(ab[:, 0] * ap[:, 1] - ab[:, 1] * ap[:, 0])
    return cross / (np.linalg.norm(ab, axis=1) + EPSILON)


def _compute_joint_angles(df: pd.DataFrame, configs: list[dict]) -> pd.DataFrame:
    """Vertex angle features. Only 2D mode is supported (pose-type is a screen-plane distinction)."""
    result: dict[str, np.ndarray] = {}
    for cfg in configs:
        result[cfg["name"]] = _angle_at_vertex_2d(
            _xy(df, cfg["joints"][0]), _xy(df, cfg["joints"][1]), _xy(df, cfg["joints"][2])
        )
    return pd.DataFrame(result, index=df.index)


def _compute_spatial_distances(df: pd.DataFrame, configs: list[dict]) -> pd.DataFrame:
    """Normalised 2D euclidean distance features."""
    result: dict[str, np.ndarray] = {}
    for cfg in configs:
        dist = _euclidean_distance_2d(_xy(df, cfg["joints"][0]), _xy(df, cfg["joints"][1]))
        if "normalization_factor" in cfg:
            nf = cfg["normalization_factor"]
            scale = _euclidean_distance_2d(_xy(df, nf[0]), _xy(df, nf[1]))
            dist = dist / (scale + EPSILON)
        result[cfg["name"]] = dist
    return pd.DataFrame(result, index=df.index)


def _midpoint(df: pd.DataFrame, idx_a: int, idx_b: int) -> np.ndarray:
    return (_xy(df, idx_a) + _xy(df, idx_b)) / 2.0


def _compute_alignment_offsets(df: pd.DataFrame, configs: list[dict]) -> pd.DataFrame:
    """
    Alignment/orientation offset features. Supported `compute` types:

        vertical_delta       -- |y_b - y_a| between two joints (2 joints).
                                 Used for arm-elevation features.

        perpendicular         -- perpendicular distance of the middle joint from the
                                 line through the outer two joints (3 joints).
                                 Used to see how far the hip sits off the shoulder-ankle line.

        vector_incline_angle  -- angle, in degrees, between a joint-to-joint (or
                                 midpoint-to-midpoint) vector and the vertical axis.
                                 0 = perfectly upright, 90 = perfectly horizontal.
                                 This is the primary mountain/warrior2 (upright) vs
                                 plank (horizontal) discriminator.
                                 2 joints -> direct vector between them.
                                 4 joints -> vector between midpoint(j0,j1) and midpoint(j2,j3).

    Raises ValueError for any unrecognised `compute` value.
    """
    result: dict[str, np.ndarray] = {}

    for cfg in configs:
        name = cfg["name"]
        compute = cfg.get("compute", "").strip().lower()
        joints = cfg["joints"]

        if compute == "vertical_delta":
            vals = np.abs(_col(df, "y", joints[1]) - _col(df, "y", joints[0]))

        elif compute == "perpendicular":
            vals = _point_to_line_distance_2d(
                _xy(df, joints[1]), _xy(df, joints[0]), _xy(df, joints[2])
            )

        elif compute == "vector_incline_angle":
            if len(joints) == 2:
                p0, p1 = _xy(df, joints[0]), _xy(df, joints[1])
            elif len(joints) == 4:
                p0 = _midpoint(df, joints[0], joints[1])
                p1 = _midpoint(df, joints[2], joints[3])
            else:
                raise ValueError(
                    f"'vector_incline_angle' feature '{name}' needs 2 or 4 joints, "
                    f"got {len(joints)}."
                )
            delta = p1 - p0
            # arctan2(|dx|, |dy|): 0 deg when the vector is vertical, 90 deg when horizontal.
            vals = np.degrees(np.arctan2(np.abs(delta[:, 0]), np.abs(delta[:, 1]) + EPSILON))

        else:
            raise ValueError(
                f"Alignment offset '{name}' has unsupported compute type '{compute}'. "
                f"Expected one of: vertical_delta, perpendicular, vector_incline_angle."
            )

        if "normalization_factor" in cfg:
            nf = cfg["normalization_factor"]
            scale = _euclidean_distance_2d(_xy(df, nf[0]), _xy(df, nf[1]))
            vals = vals / (scale + EPSILON)

        result[name] = vals

    return pd.DataFrame(result, index=df.index)


def _resolve_output_path(output_dir: str | None) -> str:
    if output_dir is not None:
        save_dir = os.path.abspath(output_dir)
    else:
        save_dir = os.path.abspath(os.path.join("..", "..", "data", "processed", "master_pose"))
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, "master_pose_features.csv")


def generate_master_features(
    yaml_path: str,
    sources: dict[str, str],
    output_dir: str | None = None,
) -> str:
    """
    Build the master pose-type training set: load every pose's raw CSV, standardize
    schemas, compute the shared engineered feature set, and save one combined CSV.

    Args:
        yaml_path: Path to master_pose.yaml.
        sources:   {pose_name: raw_csv_path} for every pose the master model must recognise.
        output_dir: Optional output directory override.

    Returns:
        Absolute path of the saved features CSV.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML config not found: '{yaml_path}'")

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    feat_config = config.get("features_config", {}).get("engineered_features", {})
    if not feat_config:
        raise ValueError("Missing 'features_config.engineered_features' block in YAML.")

    df_raw = build_master_raw_dataset(sources)

    required_indices = _collect_required_indices(feat_config)
    _validate_landmark_columns(df_raw, required_indices)

    present_meta = [col for col in METADATA_COLS if col in df_raw.columns]
    feature_frames: list[pd.DataFrame] = [df_raw[present_meta].copy()]

    if feat_config.get("joint_angles"):
        feature_frames.append(_compute_joint_angles(df_raw, feat_config["joint_angles"]))

    if feat_config.get("spatial_distances"):
        feature_frames.append(_compute_spatial_distances(df_raw, feat_config["spatial_distances"]))

    if feat_config.get("alignment_offsets"):
        feature_frames.append(_compute_alignment_offsets(df_raw, feat_config["alignment_offsets"]))

    df_features = pd.concat(feature_frames, axis=1)

    output_path = _resolve_output_path(output_dir)
    df_features.to_csv(output_path, index=False)

    print(
        f"Saved master features CSV: {len(df_features)} rows x {len(df_features.columns)} columns"
        f"\n  -> {output_path}"
        f"\n  label distribution:\n{df_features['label'].value_counts().to_string()}"
    )
    return output_path


if __name__ == "__main__":
    generate_master_features(
        yaml_path=YAML_FILE,
        sources=RAW_SOURCES,
        output_dir=OUTPUT_DIR,
    )