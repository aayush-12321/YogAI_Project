"""
Engineered feature extraction for Chair Pose (Utkatasana).

Side-view recordings only — all features resolve to the camera-facing
(reliable) side via is_left_facing using semantic joint roles.
"""

import os
import numpy as np
import pandas as pd
import yaml

# - Configuration --
YAML_FILE  = "../../configs/poses/chair_pose.yaml"
RAW_CSV    = "../../data/annotations/chair_pose/chair_pose_raw.csv"
OUTPUT_DIR = None   # None --> auto-resolves to data/processed/engineered_features/

METADATA_COLS = ["source_id", "label", "frame_number", "is_left_facing"]
EPSILON       = 1e-6

# - Orientation map --
# Maps semantic role --> (right-facing index, left-facing index).
# right-facing: person faces right, LEFT side faces camera --> left landmarks visible.
# left-facing:  person faces left, RIGHT side faces camera --> right landmarks visible.
#
# camera_* roles always resolve to the camera-facing (visible) side.
_ORIENTATION_MAP: dict[str, tuple[int, int]] = {
    "camera_shoulder": (11, 12),  # right-facing-->LEFT_SHOULDER, left-facing-->RIGHT_SHOULDER
    "camera_wrist":    (15, 16),
    "camera_hip":      (23, 24),
    "camera_knee":     (25, 26),
    "camera_ankle":    (27, 28),
}


def _resolve(joint: int | str, is_left: bool) -> int:
    """Resolve a raw index or semantic role to a concrete MediaPipe index."""
    if isinstance(joint, str):
        if joint not in _ORIENTATION_MAP:
            raise ValueError(f"Unknown semantic role '{joint}'. Valid: {sorted(_ORIENTATION_MAP)}")
        right_idx, left_idx = _ORIENTATION_MAP[joint]
        return left_idx if is_left else right_idx
    return int(joint)


# - Coordinate helpers -
def _xy(df, idx):  return df[[f"x_{idx}", f"y_{idx}"]].to_numpy(dtype=np.float64)
def _xyz(df, idx): return df[[f"x_{idx}", f"y_{idx}", f"z_{idx}"]].to_numpy(dtype=np.float64)
def _cx(df, idx):  return df[f"x_{idx}"].to_numpy(dtype=np.float64)
def _cy(df, idx):  return df[f"y_{idx}"].to_numpy(dtype=np.float64)
def _dist2d(a, b): return np.linalg.norm(a - b, axis=1)


# - Core math -
def _angle_2d(a, b, c):
    """Angle at vertex B in degrees using X/Y only (avoids depth jitter)."""
    ba, bc = a - b, c - b
    cos = np.clip(
        np.einsum("ij,ij->i", ba, bc) /
        (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + EPSILON),
        -1.0, 1.0,
    )
    return np.degrees(np.arccos(cos))


def _angle_3d(a, b, c):
    """Angle at vertex B in degrees using X/Y/Z (for genuine 3D flexion)."""
    ba, bc = a - b, c - b
    cos = np.clip(
        np.einsum("ij,ij->i", ba, bc) /
        (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + EPSILON),
        -1.0, 1.0,
    )
    return np.degrees(np.arccos(cos))


# - Feature computation 
def _compute_features(df: pd.DataFrame, feat_config: dict) -> pd.DataFrame:
    """
    Compute all engineered features with orientation-aware index resolution.

    Splits rows into right-facing / left-facing subsets, resolves semantic
    joint roles for each, then recombines — ensuring every feature always
    reads from the visible camera-side landmark.
    """
    is_left = df["is_left_facing"].to_numpy(dtype=bool)
    out     = np.empty((len(df), 0))   # grows column-by-column
    names   = []

    all_configs = (
        [("angle",    c) for c in feat_config.get("joint_angles",       [])] +
        [("distance", c) for c in feat_config.get("spatial_distances",  [])] +
        [("offset",   c) for c in feat_config.get("alignment_offsets",  [])]
    )

    result: dict[str, np.ndarray] = {}

    for kind, cfg in all_configs:
        name    = cfg["name"]
        compute = cfg.get("compute", "").strip().lower()
        col     = np.empty(len(df))

        for left_flag in (False, True):
            mask = is_left == left_flag
            if not mask.any():
                continue
            sub = df[mask]

            # Resolve joints for this orientation
            j = [_resolve(x, left_flag) for x in cfg["joints"]]

            if kind == "angle":
                if cfg.get("mode") == "2d":
                    vals = _angle_2d(_xy(sub, j[0]), _xy(sub, j[1]), _xy(sub, j[2]))
                else:
                    vals = _angle_3d(_xyz(sub, j[0]), _xyz(sub, j[1]), _xyz(sub, j[2]))

            elif compute == "vertical_delta":
                # |y_b - y_a|  (MediaPipe Y increases downward;
                # wrist above shoulder --> smaller Y --> large delta after abs)
                vals = np.abs(_cy(sub, j[1]) - _cy(sub, j[0]))

            elif compute == "lateral_delta":
                # |x_a - x_b|  — horizontal separation on screen
                vals = np.abs(_cx(sub, j[0]) - _cx(sub, j[1]))

            else:
                raise ValueError(
                    f"Feature '{name}' has unsupported compute type '{compute}'. "
                    f"Expected: vertical_delta | lateral_delta."
                )

            # Normalize distance features by torso length (shoulder-->hip)
            if "normalization_factor" in cfg:
                nf    = [_resolve(x, left_flag) for x in cfg["normalization_factor"]]
                scale = _dist2d(_xy(sub, nf[0]), _xy(sub, nf[1]))
                vals  = vals / (scale + EPSILON)

            col[mask] = vals

        result[name] = col

    return pd.DataFrame(result, index=df.index)


# - Path helpers 
def _resolve_output_path(raw_csv_path: str, output_dir: str | None) -> str:
    base     = os.path.basename(raw_csv_path)
    out_name = (base[:-len("_raw.csv")] + "_features.csv"
                if base.endswith("_raw.csv")
                else os.path.splitext(base)[0] + "_features.csv")

    if output_dir:
        save_dir = os.path.abspath(output_dir)
    else:
        search = os.path.abspath(os.path.dirname(raw_csv_path))
        root   = search
        for _ in range(6):
            if os.path.basename(search) == "data":
                root = os.path.dirname(search)
                break
            parent = os.path.dirname(search)
            if parent == search:
                break
            search = parent
        save_dir = os.path.join(root, "data", "processed", "engineered_features")

    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, out_name)


# - Public entry point -
def generate_engineered_features(
    yaml_path: str,
    raw_csv_path: str,
    output_dir: str | None = None,
) -> str:
    """
    Load raw landmarks CSV, compute chair-pose features, save features CSV.

    Args:
        yaml_path:    Path to chair_pose.yaml.
        raw_csv_path: Path to *_raw.csv from extract_landmarks_chair.py.
        output_dir:   Optional output folder; auto-resolved when None.

    Returns:
        Absolute path of the saved features CSV.
    """
    for path in (yaml_path, raw_csv_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: '{path}'")

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    feat_config = config.get("features_config", {}).get("engineered_features", {})
    if not feat_config:
        raise ValueError("Missing 'features_config.engineered_features' in YAML.")

    df_raw = pd.read_csv(raw_csv_path)
    print(f"Loaded {len(df_raw)} rows from '{raw_csv_path}'.")

    if "is_left_facing" not in df_raw.columns:
        raise ValueError(
            "'is_left_facing' column missing — re-run extract_landmarks_chair.py."
        )

    n_left  = int(df_raw["is_left_facing"].sum())
    n_right = len(df_raw) - n_left
    print(f"Orientation: {n_right} right-facing, {n_left} left-facing rows.")

    meta        = [c for c in METADATA_COLS if c in df_raw.columns]
    df_features = pd.concat(
        [df_raw[meta].copy(), _compute_features(df_raw, feat_config)],
        axis=1,
    )

    out_path = _resolve_output_path(raw_csv_path, output_dir)
    df_features.to_csv(out_path, index=False)
    print(f"Saved {len(df_features)} rows × {len(df_features.columns)} cols --> '{out_path}'")
    return out_path


# CLI 
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate engineered features for Chair Pose (Utkatasana)."
    )
    parser.add_argument("--yaml",    "-y", default=YAML_FILE)
    parser.add_argument("--raw-csv", "-r", default=RAW_CSV)
    parser.add_argument("--output",  "-o", default=None)
    args = parser.parse_args()

    generate_engineered_features(args.yaml, args.raw_csv, args.output)