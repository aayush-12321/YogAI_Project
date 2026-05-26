import os

import numpy as np
import pandas as pd
import yaml

YAML_FILE = "../../configs/poses/mountain_pose.yaml"
RAW_CSV = "../../data/annotations/mountain_pose/mountain_pose_raw.csv"
OUTPUT_DIR = None  # Set to a string path to override; None uses default logic


METADATA_COLS = ["source_id", "label", "frame_number"]
EPSILON = 1e-6


def _validate_inputs(yaml_path: str, raw_csv_path: str) -> None:
    """Raise with a clear message if either required input file is missing."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML config not found: '{yaml_path}'")
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw landmarks CSV not found: '{raw_csv_path}'")


def _validate_landmark_columns(df: pd.DataFrame, required_indices: set[int]) -> None:
    """
    Confirm every landmark index needed by the feature config has x/y/z columns
    in the dataframe. Raises ValueError listing all missing columns at once.
    """
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
    """Return the set of all landmark indices referenced anywhere in the feature config."""
    indices: set[int] = set()

    for cfg in feat_config.get("joint_angles", []):
        indices.update(cfg["joints"])

    for cfg in feat_config.get("spatial_distances", []):
        indices.update(cfg["joints"])
        if "normalization_factor" in cfg:
            indices.update(cfg["normalization_factor"])

    for cfg in feat_config.get("alignment_offsets", []):
        indices.update(cfg["joints"])
        if "normalization_factor" in cfg:
            indices.update(cfg["normalization_factor"])

    return indices


def _xyz(df: pd.DataFrame, idx: int) -> np.ndarray:
    """Return an (N, 3) array of x/y/z coordinates for landmark index idx."""
    return df[[f"x_{idx}", f"y_{idx}", f"z_{idx}"]].to_numpy(dtype=np.float64)


def _angle_at_vertex(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Compute the 3D angle at vertex B for each row, in degrees.

    Args:
        a: (N, 3) array for point A.
        b: (N, 3) array for vertex B.
        c: (N, 3) array for point C.

    Returns:
        (N,) array of angles in degrees.
    """
    ba = a - b
    bc = c - b
    dot = np.einsum("ij,ij->i", ba, bc)
    norm_ba = np.linalg.norm(ba, axis=1)
    norm_bc = np.linalg.norm(bc, axis=1)
    cosine = np.clip(dot / (norm_ba * norm_bc + EPSILON), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return (N,) array of Euclidean distances between paired rows of a and b."""
    return np.linalg.norm(a - b, axis=1)


def _point_to_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute perpendicular distance from each point P to the line through A and B.

    Returns:
        (N,) array of distances.
    """
    ab = b - a
    ap = p - a
    cross = np.cross(ap, ab)
    return np.linalg.norm(cross, axis=1) / (np.linalg.norm(ab, axis=1) + EPSILON)


def _compute_joint_angles(df: pd.DataFrame, configs: list[dict]) -> pd.DataFrame:
    """
    Vectorized computation of all joint angle features.

    Each config entry must have:
        name   : output column name
        joints : [idx_a, idx_b, idx_c]  -- angle measured at idx_b
    """
    result = {}
    for cfg in configs:
        j = cfg["joints"]
        result[cfg["name"]] = _angle_at_vertex(_xyz(df, j[0]), _xyz(df, j[1]), _xyz(df, j[2]))
    return pd.DataFrame(result, index=df.index)


def _compute_spatial_distances(df: pd.DataFrame, configs: list[dict]) -> pd.DataFrame:
    """
    Vectorized computation of all spatial distance features.

    Each config entry must have:
        name                 : output column name
        joints               : [idx_a, idx_b]
        normalization_factor : [idx_c, idx_d]  (optional) -- divides distance by c-d distance
    """
    result = {}
    for cfg in configs:
        j = cfg["joints"]
        dist = _euclidean_distance(_xyz(df, j[0]), _xyz(df, j[1]))

        if "normalization_factor" in cfg:
            nf = cfg["normalization_factor"]
            norm_dist = _euclidean_distance(_xyz(df, nf[0]), _xyz(df, nf[1]))
            dist = dist / (norm_dist + EPSILON)

        result[cfg["name"]] = dist
    return pd.DataFrame(result, index=df.index)


def _compute_alignment_offsets(df: pd.DataFrame, configs: list[dict]) -> pd.DataFrame:
    """
    Vectorized computation of alignment offset features.

    Supported offset types are inferred from joint count and are fully data-driven
    -- no hardcoded pose names:

        2 joints : absolute lateral X-axis delta between the two points.
        3 joints : perpendicular distance of the middle point from the line
                   connecting the first and third points (e.g. hip sag/pike).
        4 joints, slope-delta pattern (ears+shoulders, arms):
                   absolute difference in line slopes between pair [0,1] and pair [2,3].
        4 joints, plumb-line pattern (spine chain):
                   sum of perpendicular deviations of inner joints [1,2] from the
                   line connecting the outer joints [0,3].

    The distinction between the two 4-joint patterns is made by whether the name
    contains "plumb" -- a lightweight convention that keeps the YAML config readable
    without adding an explicit 'type' field to every entry.
    """
    result = {}
    for cfg in configs:
        j = cfg["joints"]
        name = cfg["name"]
        n = len(j)

        try:
            if n == 2:
                # Absolute lateral delta between two points
                vals = np.abs(
                    df[f"x_{j[0]}"].to_numpy(dtype=np.float64)
                    - df[f"x_{j[1]}"].to_numpy(dtype=np.float64)
                )

            elif n == 3:
                # Perpendicular deviation of the middle joint from the outer-joint line
                vals = _point_to_line_distance(
                    _xyz(df, j[1]), _xyz(df, j[0]), _xyz(df, j[2])
                )

            elif n == 4 and "plumb" in name.lower():
                # Spine/plumb-line: sum of perpendicular deviations of j[1] and j[2]
                # from the line j[0]->j[3]
                outer_a = _xyz(df, j[0])
                outer_b = _xyz(df, j[3])
                dev_1 = _point_to_line_distance(_xyz(df, j[1]), outer_a, outer_b)
                dev_2 = _point_to_line_distance(_xyz(df, j[2]), outer_a, outer_b)
                vals = dev_1 + dev_2

            elif n == 4:
                # Slope-delta: difference between line angles of pair [0,1] and [2,3]
                angle_01 = np.arctan2(
                    df[f"y_{j[1]}"].to_numpy(dtype=np.float64) - df[f"y_{j[0]}"].to_numpy(dtype=np.float64),
                    df[f"x_{j[1]}"].to_numpy(dtype=np.float64) - df[f"x_{j[0]}"].to_numpy(dtype=np.float64),
                )
                angle_23 = np.arctan2(
                    df[f"y_{j[3]}"].to_numpy(dtype=np.float64) - df[f"y_{j[2]}"].to_numpy(dtype=np.float64),
                    df[f"x_{j[3]}"].to_numpy(dtype=np.float64) - df[f"x_{j[2]}"].to_numpy(dtype=np.float64),
                )
                
                # 1. Compute raw difference in radians
                delta_rad = angle_01 - angle_23
                
                # 2. Map the delta back into the accurate range [-pi, +pi] to cancel boundary wrap-around
                delta_rad = (delta_rad + np.pi) % (2 * np.pi) - np.pi
                
                # 3. Take the absolute degrees (now safely bounded between 0° and 180°)
                vals = np.degrees(np.abs(delta_rad))

            else:
                raise ValueError(
                    f"Alignment offset '{name}' has {n} joints; expected 2, 3, or 4."
                )

        except Exception as exc:
            # Assign NaN so downstream imputation/drop can handle it explicitly
            print(f"Warning: failed computing alignment offset '{name}': {exc}")
            vals = np.full(len(df), np.nan)

        if "normalization_factor" in cfg:
            nf = cfg["normalization_factor"]
            scale = _euclidean_distance(_xyz(df, nf[0]), _xyz(df, nf[1]))
            vals = vals / (scale + EPSILON)

        result[name] = vals

    return pd.DataFrame(result, index=df.index)


def _resolve_output_path(raw_csv_path: str, output_dir: str | None) -> str:
    """
    Derive the full output CSV path.

    If output_dir is provided, saves there. Otherwise saves to
    <project_root>/data/processed/engineered_features/, where project_root is
    inferred by walking up from the raw CSV's location until the 'data' directory
    is found -- matching the same convention used by the landmark extractor.
    """
    base = os.path.basename(raw_csv_path)

    if base.endswith("_raw.csv"):
        out_filename = base[: -len("_raw.csv")] + "_features.csv"
    else:
        name, _ = os.path.splitext(base)
        out_filename = f"{name}_features.csv"

    if output_dir is not None:
        save_dir = os.path.abspath(output_dir)
    else:
        # Walk up from the CSV location to find the 'data' directory
        search = os.path.abspath(os.path.dirname(raw_csv_path))
        project_root = search
        for _ in range(6):  # cap traversal depth
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
    Compute engineered geometric features from a raw landmark CSV using a pose YAML config.

    Args:
        yaml_path:    Path to the pose YAML file containing 'features_config.engineered_features'.
        raw_csv_path: Path to the raw landmarks CSV produced by the landmark extractor.
        output_dir:   Optional directory to save the output CSV. When None, saves to
                      <project_root>/data/processed/engineered_features/.

    Returns:
        Absolute path to the saved features CSV.

    Raises:
        FileNotFoundError: If either input file does not exist.
        ValueError: If the YAML has no engineered_features block, or if required
                    landmark columns are missing from the CSV.
    """
    _validate_inputs(yaml_path, raw_csv_path)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    feat_config = config.get("features_config", {}).get("engineered_features", {})
    if not feat_config:
        raise ValueError("Missing 'features_config.engineered_features' block in YAML.")

    #  Startup normalization map 
    _normalized_features: list[str] = []
    for section in ("spatial_distances", "alignment_offsets"):
        for cfg in feat_config.get(section, []):
            if "normalization_factor" in cfg:
                nf = cfg["normalization_factor"]
                _normalized_features.append(
                    f"  [{section}] '{cfg['name']}' ÷ dist(landmark {nf[0]}, landmark {nf[1]})"
                )
    if _normalized_features:
        print("[FeatureEngineering] Dynamic scale-normalization ARMED for:")
        print("\n".join(_normalized_features))
    else:
        print("[FeatureEngineering] No normalization_factor entries found -- all features are raw.")
    # 

    df_raw = pd.read_csv(raw_csv_path)
    print(f"Loaded '{raw_csv_path}' -- {len(df_raw)} rows, {len(df_raw.columns)} columns.")

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