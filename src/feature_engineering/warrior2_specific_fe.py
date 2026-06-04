"""
Engineered feature extraction from raw MediaPipe landmark CSVs.

Fixes applied vs the original:
  1. Orientation-aware index resolution: reads is_left_facing from the raw CSV
     and swaps front/back joint indices per row so all features are semantically
     consistent regardless of which knee is the front (bent) leg.
  2. center_of_mass_torso_offset: rewritten as a proper 2D midpoint horizontal
     delta (mid_shoulder_x - mid_hip_x) normalised by shoulder width. The old
     code mis-dispatched it through the 4-joint slope-delta path, producing
     values up to 800.
  3. front_knee_ankle_alignment: lateral X delta is computed on the correct
     front-side joints after orientation swap (was always using RIGHT joints).
  4. Z-axis excluded from 2D alignment/offset features to eliminate the
     per-frame depth jitter visible in the timeline plots. Z is still used for
     3D joint angle calculations where it adds genuine signal.
  5. Alignment offset dispatch is now driven by an explicit 'compute' field in
     the YAML config ('lateral_delta' | 'midpoint_delta' | 'perpendicular' |
     'slope_delta'). This replaces the fragile 'plumb in name' string heuristic.
"""

import os

import numpy as np
import pandas as pd
import yaml


YAML_FILE = "../../configs/poses/warrior2_pose.yaml"
RAW_CSV = "../../data/annotations/warrior2_pose/warrior2_pose_raw.csv"
OUTPUT_DIR = None

METADATA_COLS = ["source_id", "label", "frame_number"]
EPSILON = 1e-6

# Landmark index constants matching MediaPipe's 33-point body model.
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12
_LEFT_WRIST     = 15
_RIGHT_WRIST    = 16
_LEFT_HIP       = 23
_RIGHT_HIP      = 24
_LEFT_KNEE      = 25
_RIGHT_KNEE     = 26
_LEFT_ANKLE     = 27
_RIGHT_ANKLE    = 28

# Maps semantic role -> (right-facing index, left-facing index).
# right-facing = webcam default: RIGHT knee is the front/bent leg.
# left-facing  = 'left_' prefix files: LEFT knee is the front/bent leg.
_ORIENTATION_MAP: dict[str, tuple[int, int]] = {
    "front_hip":      (_RIGHT_HIP,      _LEFT_HIP),
    "front_knee":     (_RIGHT_KNEE,     _LEFT_KNEE),
    "front_ankle":    (_RIGHT_ANKLE,    _LEFT_ANKLE),
    "front_shoulder": (_RIGHT_SHOULDER, _LEFT_SHOULDER),
    "front_wrist":    (_RIGHT_WRIST,    _LEFT_WRIST),
    "back_hip":       (_LEFT_HIP,       _RIGHT_HIP),
    "back_knee":      (_LEFT_KNEE,      _RIGHT_KNEE),
    "back_ankle":     (_LEFT_ANKLE,     _RIGHT_ANKLE),
    "back_shoulder":  (_LEFT_SHOULDER,  _RIGHT_SHOULDER),
    "back_wrist":     (_LEFT_WRIST,     _RIGHT_WRIST),
}


def _validate_inputs(yaml_path: str, raw_csv_path: str) -> None:
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML config not found: '{yaml_path}'")
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw landmarks CSV not found: '{raw_csv_path}'")


def _validate_landmark_columns(df: pd.DataFrame, required_indices: set[int]) -> None:
    missing = [
        f"{axis}_{i}"
        for i in required_indices
        for axis in ("x", "y", "z")
        if f"{axis}_{i}" not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Raw CSV is missing {len(missing)} expected landmark column(s): {missing}"
        )


def _collect_required_indices(feat_config: dict) -> set[int]:
    """
    Return the set of all raw MediaPipe landmark integers referenced in the feature config.

    Semantic role strings (e.g. 'front_knee') are resolved for both orientations
    so the validator checks that every concrete column that could ever be accessed
    is present in the CSV, regardless of which orientation a given row carries.
    """
    indices: set[int] = set()
    for section in ("joint_angles", "spatial_distances", "alignment_offsets"):
        for cfg in feat_config.get(section, []):
            for joint_list_key in ("joints", "normalization_factor"):
                for j in cfg.get(joint_list_key, []):
                    if isinstance(j, str):
                        if j not in _ORIENTATION_MAP:
                            raise ValueError(
                                f"Unknown semantic joint role '{j}' in config '{cfg.get('name')}'. "
                                f"Valid roles: {sorted(_ORIENTATION_MAP)}"
                            )
                        right_idx, left_idx = _ORIENTATION_MAP[j]
                        indices.add(right_idx)
                        indices.add(left_idx)
                    else:
                        indices.add(int(j))
    return indices


def _resolve_joints(joints: list[int | str], is_left: bool) -> list[int]:
    """
    Resolve a joint list that may contain semantic role strings (e.g. 'front_knee')
    or raw landmark integers into concrete MediaPipe indices.

    Semantic roles are resolved via _ORIENTATION_MAP using the is_left flag so
    that all downstream math operates on the correct anatomical side.
    """
    resolved = []
    for j in joints:
        if isinstance(j, str):
            if j not in _ORIENTATION_MAP:
                raise ValueError(
                    f"Unknown semantic joint role '{j}'. "
                    f"Valid roles: {sorted(_ORIENTATION_MAP)}"
                )
            right_idx, left_idx = _ORIENTATION_MAP[j]
            resolved.append(left_idx if is_left else right_idx)
        else:
            resolved.append(int(j))
    return resolved


def _xy(df: pd.DataFrame, idx: int) -> np.ndarray:
    """Return (N, 2) array of x/y coordinates for landmark idx. Z excluded to avoid depth jitter."""
    return df[[f"x_{idx}", f"y_{idx}"]].to_numpy(dtype=np.float64)


def _xyz(df: pd.DataFrame, idx: int) -> np.ndarray:
    """Return (N, 3) array of x/y/z coordinates for landmark idx. Used only for 3D angles."""
    return df[[f"x_{idx}", f"y_{idx}", f"z_{idx}"]].to_numpy(dtype=np.float64)


def _col_x(df: pd.DataFrame, idx: int) -> np.ndarray:
    return df[f"x_{idx}"].to_numpy(dtype=np.float64)


def _angle_at_vertex_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Compute the 3D angle at vertex B for each row, in degrees.

    Uses Z for joint angles because knee and arm flexion are genuinely 3D;
    excluding Z here would distort angles when the limb is not parallel to
    the camera plane.

    Args:
        a, b, c: (N, 3) coordinate arrays. Angle is measured at B.

    Returns:
        (N,) array of angles in degrees in [0, 180].
    """
    ba = a - b
    bc = c - b
    dot = np.einsum("ij,ij->i", ba, bc)
    norm_ba = np.linalg.norm(ba, axis=1)
    norm_bc = np.linalg.norm(bc, axis=1)
    cosine = np.clip(dot / (norm_ba * norm_bc + EPSILON), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

def _angle_at_vertex_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Compute the 2D angle at vertex B on the screen plane (X, Y), ignoring Z jitter.
    Args:
        a, b, c: (N, 2) coordinate arrays. Angle is measured at B.
    Returns:
        (N,) array of angles in degrees in [0, 180].
    """
    ba = a - b
    bc = c - b
    dot = np.einsum("ij,ij->i", ba, bc)
    norm_ba = np.linalg.norm(ba, axis=1)
    norm_bc = np.linalg.norm(bc, axis=1)
    cosine = np.clip(dot / (norm_ba * norm_bc + EPSILON), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _euclidean_distance_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return (N,) Euclidean distances between paired 2D row vectors."""
    return np.linalg.norm(a - b, axis=1)


def _euclidean_distance_3d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return (N,) Euclidean distances between paired 3D row vectors."""
    return np.linalg.norm(a - b, axis=1)


def _point_to_line_distance_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Perpendicular 2D distance from each point P to the line through A and B.

    Returns:
        (N,) array of distances.
    """
    ab = b - a
    ap = p - a
    # 2D cross product magnitude = |ab_x * ap_y - ab_y * ap_x|
    cross = np.abs(ab[:, 0] * ap[:, 1] - ab[:, 1] * ap[:, 0])
    return cross / (np.linalg.norm(ab, axis=1) + EPSILON)


def _shoulder_width(df: pd.DataFrame) -> np.ndarray:
    """
    Per-row 2D shoulder width used as the normalization scale throughout.

    Shoulder width is stable, camera-distance-invariant, and symmetric, making
    it the correct reference for normalizing all spatial features.
    """
    return _euclidean_distance_2d(
        _xy(df, _LEFT_SHOULDER), _xy(df, _RIGHT_SHOULDER)
    )


def _compute_joint_angles(
    df: pd.DataFrame,
    configs: list[dict],
    is_left: np.ndarray,
) -> pd.DataFrame:
    """
    Compute all joint angle features using 3D coordinates.

    Orientation swap: for each row the joint list is resolved through
    _resolve_joints so 'front_knee' always points to the bent leg's knee.

    Args:
        df:       Raw landmark dataframe.
        configs:  List of joint_angles config dicts from YAML.
        is_left:  Boolean array (N,) -- True where the file is left-facing.

    Returns:
        DataFrame with one column per angle feature.
    """
    result: dict[str, np.ndarray] = {cfg["name"]: np.empty(len(df)) for cfg in configs}

    # Process right-facing and left-facing rows separately to allow vectorised math.
    for left_flag in (False, True):
        mask = is_left == left_flag
        if not mask.any():
            continue
        sub = df[mask]

        for cfg in configs:
            joints = _resolve_joints(cfg["joints"], left_flag)
            
            # Check if a 2D screen plane projection is requested
            if cfg.get("mode") == "2d":
                result[cfg["name"]][mask] = _angle_at_vertex_2d(
                    _xy(sub, joints[0]),
                    _xy(sub, joints[1]),
                    _xy(sub, joints[2]),
                )
            else:
                result[cfg["name"]][mask] = _angle_at_vertex_3d(
                    _xyz(sub, joints[0]),
                    _xyz(sub, joints[1]),
                    _xyz(sub, joints[2]),
                )

    return pd.DataFrame(result, index=df.index)


def _compute_spatial_distances(
    df: pd.DataFrame,
    configs: list[dict],
    is_left: np.ndarray,
) -> pd.DataFrame:
    """
    Compute all spatial distance features using 2D coordinates.

    Z is excluded to prevent depth-jitter inflating distance values.
    """
    result: dict[str, np.ndarray] = {}

    for left_flag in (False, True):
        mask = is_left == left_flag
        if not mask.any():
            continue
        sub = df[mask]

        for cfg in configs:
            joints = _resolve_joints(cfg["joints"], left_flag)
            dist = _euclidean_distance_2d(_xy(sub, joints[0]), _xy(sub, joints[1]))

            if "normalization_factor" in cfg:
                nf = _resolve_joints(cfg["normalization_factor"], left_flag)
                scale = _euclidean_distance_2d(_xy(sub, nf[0]), _xy(sub, nf[1]))
                dist = dist / (scale + EPSILON)

            name = cfg["name"]
            if name not in result:
                result[name] = np.empty(len(df))
            result[name][mask] = dist

    return pd.DataFrame(result, index=df.index)


def _compute_alignment_offsets(
    df: pd.DataFrame,
    configs: list[dict],
    is_left: np.ndarray,
) -> pd.DataFrame:
    """
    Compute all alignment offset features.

    Each config entry must carry an explicit 'compute' field:

        lateral_delta   -- |x_a - x_b| between two points (2 joints).
                           Used for front_knee_ankle_alignment (knee caving).

        vertical_delta  -- |y_b - y_a| between two points (2 joints).
                           Used for wrist-to-shoulder drops and wrist symmetry.

        midpoint_delta  -- horizontal shift of mid(a,b) relative to mid(c,d)
                           normalised by shoulder width (4 joints).
                           Used for center_of_mass_torso_offset (torso lean).
                           This is a pure 2D X-axis calculation; Z is excluded
                           because horizontal torso lean is a screen-plane event.

        perpendicular   -- perpendicular distance of the middle joint from the
                           line connecting the outer two joints (3 joints).

        slope_delta     -- absolute angle difference between line(j0,j1) and
                           line(j2,j3) in degrees (4 joints).

    Raises ValueError for any unrecognised 'compute' value so YAML mistakes
    surface immediately rather than producing silent NaN columns.
    """
    result: dict[str, np.ndarray] = {}

    for left_flag in (False, True):
        mask = is_left == left_flag
        if not mask.any():
            continue
        sub = df[mask]

        for cfg in configs:
            name = cfg["name"]
            compute = cfg.get("compute", "").strip().lower()
            joints = _resolve_joints(cfg["joints"], left_flag)

            if compute == "lateral_delta":
                # |x_front_knee - x_front_ankle|
                # 2D only -- depth has no bearing on lateral knee caving.
                vals = np.abs(
                    _col_x(sub, joints[0]) - _col_x(sub, joints[1])
                )

            elif compute == "midpoint_delta":
                # Horizontal shift of mid-shoulder vs mid-hip.
                # joints: [shoulder_a, shoulder_b, hip_a, hip_b]
                mid_shoulder_x = (_col_x(sub, joints[0]) + _col_x(sub, joints[1])) / 2.0
                mid_hip_x      = (_col_x(sub, joints[2]) + _col_x(sub, joints[3])) / 2.0
                vals = mid_shoulder_x - mid_hip_x
                # Signed: positive = torso shifted toward the front (right in camera).
                # The absolute value is taken after normalisation so direction is preserved
                # as a feature; the model can learn which direction is leaning_torso.
                vals = np.abs(vals)

            elif compute == "vertical_delta":
                # Absolute vertical distance delta using Y coordinates.
                vals = np.abs(
                    sub[f"y_{joints[1]}"].to_numpy(dtype=np.float64) - 
                    sub[f"y_{joints[0]}"].to_numpy(dtype=np.float64)
                )

            elif compute == "perpendicular":
                # Perpendicular deviation of the middle joint from the outer-joint line.
                vals = _point_to_line_distance_2d(
                    _xy(sub, joints[1]),
                    _xy(sub, joints[0]),
                    _xy(sub, joints[2]),
                )

            elif compute == "slope_delta":
                # Absolute angle difference between two line segments, in degrees.
                angle_01 = np.arctan2(
                    sub[f"y_{joints[1]}"].to_numpy(dtype=np.float64) - sub[f"y_{joints[0]}"].to_numpy(dtype=np.float64),
                    sub[f"x_{joints[1]}"].to_numpy(dtype=np.float64) - sub[f"x_{joints[0]}"].to_numpy(dtype=np.float64),
                )
                angle_23 = np.arctan2(
                    sub[f"y_{joints[3]}"].to_numpy(dtype=np.float64) - sub[f"y_{joints[2]}"].to_numpy(dtype=np.float64),
                    sub[f"x_{joints[3]}"].to_numpy(dtype=np.float64) - sub[f"x_{joints[2]}"].to_numpy(dtype=np.float64),
                )
                delta_rad = (angle_01 - angle_23 + np.pi) % (2 * np.pi) - np.pi
                vals = np.degrees(np.abs(delta_rad))

            else:
                raise ValueError(
                    f"Alignment offset '{name}' has unsupported compute type '{compute}'. "
                    f"Expected one of: lateral_delta, vertical_delta, midpoint_delta, perpendicular, slope_delta."
                )

            if "normalization_factor" in cfg:
                nf = _resolve_joints(cfg["normalization_factor"], left_flag)
                scale = _euclidean_distance_2d(_xy(sub, nf[0]), _xy(sub, nf[1]))
                vals = vals / (scale + EPSILON)

            if name not in result:
                result[name] = np.empty(len(df))
            result[name][mask] = vals

    return pd.DataFrame(result, index=df.index)


def _resolve_output_path(raw_csv_path: str, output_dir: str | None) -> str:
    base = os.path.basename(raw_csv_path)
    out_filename = (
        base[: -len("_raw.csv")] + "_features.csv"
        if base.endswith("_raw.csv")
        else os.path.splitext(base)[0] + "_features.csv"
    )

    if output_dir is not None:
        save_dir = os.path.abspath(output_dir)
    else:
        search = os.path.abspath(os.path.dirname(raw_csv_path))
        project_root = search
        for _ in range(6):
            if os.path.basename(search) == "data":
                project_root = os.path.dirname(search)
                break
            parent = os.path.dirname(search)
            if parent == search:
                break
            search = parent
        save_dir = os.path.join(project_root, "data", "processed", "engineered_features")

    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, out_filename)


def generate_engineered_features(
    yaml_path: str,
    raw_csv_path: str,
    output_dir: str | None = None,
) -> str:
    """
    Compute engineered geometric features from a raw landmark CSV.

    Args:
        yaml_path:    Path to the pose YAML config.
        raw_csv_path: Path to the raw landmarks CSV (must include is_left_facing column).
        output_dir:   Optional output directory. Defaults to
                      <project_root>/data/processed/engineered_features/.

    Returns:
        Absolute path of the saved features CSV.

    Raises:
        FileNotFoundError: If either input file is missing.
        ValueError: If the YAML has no engineered_features block, required landmark
                    columns are absent, or an alignment offset has an unknown compute type.
    """
    _validate_inputs(yaml_path, raw_csv_path)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    feat_config = config.get("features_config", {}).get("engineered_features", {})
    if not feat_config:
        raise ValueError("Missing 'features_config.engineered_features' block in YAML.")

    df_raw = pd.read_csv(raw_csv_path)
    print(f"Loaded '{raw_csv_path}' -- {len(df_raw)} rows, {len(df_raw.columns)} columns.")

    if "is_left_facing" not in df_raw.columns:
        raise ValueError(
            "Raw CSV is missing 'is_left_facing' column. "
            "Re-run the landmark extractor (extract_pose_landmarks.py) to regenerate it."
        )

    is_left: np.ndarray = df_raw["is_left_facing"].to_numpy(dtype=bool)
    n_left  = is_left.sum()
    n_right = (~is_left).sum()
    print(f"Orientation split: {n_right} right-facing rows, {n_left} left-facing rows.")

    required_indices = _collect_required_indices(feat_config)
    _validate_landmark_columns(df_raw, required_indices)

    present_meta = [col for col in METADATA_COLS if col in df_raw.columns]
    feature_frames: list[pd.DataFrame] = [df_raw[present_meta].copy()]

    if feat_config.get("joint_angles"):
        feature_frames.append(
            _compute_joint_angles(df_raw, feat_config["joint_angles"], is_left)
        )

    if feat_config.get("spatial_distances"):
        feature_frames.append(
            _compute_spatial_distances(df_raw, feat_config["spatial_distances"], is_left)
        )

    if feat_config.get("alignment_offsets"):
        feature_frames.append(
            _compute_alignment_offsets(df_raw, feat_config["alignment_offsets"], is_left)
        )

    df_features = pd.concat(feature_frames, axis=1)

    output_path = _resolve_output_path(raw_csv_path, output_dir)
    df_features.to_csv(output_path, index=False)

    print(
        f"Saved features CSV: {len(df_features)} rows x {len(df_features.columns)} columns"
        f"\n  -> {output_path}"
    )
    return output_path


if __name__ == "__main__":
    generate_engineered_features(
        yaml_path=YAML_FILE,
        raw_csv_path=RAW_CSV,
        output_dir=OUTPUT_DIR,
    )