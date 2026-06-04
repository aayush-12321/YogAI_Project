import os
import numpy as np
import pandas as pd
import yaml

# File Path Configuration
YAML_FILE = "configs/poses/plank_pose.yaml"
RAW_CSV = "data/annotations/plank_pose/plank_pose_raw.csv"
OUTPUT_DIR = None

EPSILON = 1e-6


def _validate_inputs(yaml_path: str, raw_csv_path: str) -> None:
    """Confirm files exist before starting processing pipeline."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML config not found: '{yaml_path}'")
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw landmarks CSV not found: '{raw_csv_path}'")


def _validate_landmark_columns(df: pd.DataFrame, required_names: set[str]) -> None:
    """Validates that your text-named coordinate columns exist in the dataframe."""
    missing = [
        f"{joint}_{axis}"
        for joint in required_names
        for axis in ("x", "y", "z")
        if f"{joint}_{axis}" not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Raw CSV is missing {len(missing)} expected landmark column(s): {missing}"
        )


def _collect_required_indices(feat_config: dict) -> set[str]:
    """Scans config block for all named text joint keys."""
    indices: set[str] = set()
    for cfg in feat_config.get("joint_angles", []):
        indices.update(cfg["joints"])
    return indices


def _xyz(df: pd.DataFrame, joint_name: str) -> np.ndarray:
    """Loads text-named columns into an (N, 3) numpy matrix layout."""
    return df[[f"{joint_name}_x", f"{joint_name}_y", f"{joint_name}_z"]].to_numpy(dtype=np.float64)


def _angle_at_vertex(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Computes continuous hinge 3D angles at vertex point B (in degrees)."""
    ba = a - b
    bc = c - b
    dot = np.einsum("ij,ij->i", ba, bc)
    norm_ba = np.linalg.norm(ba, axis=1)
    norm_bc = np.linalg.norm(bc, axis=1)
    cosine = np.clip(dot / (norm_ba * norm_bc + EPSILON), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _compute_joint_angles(df: pd.DataFrame, configs: list[dict]) -> pd.DataFrame:
    """Loops through joint entries to compute vector mathematical features."""
    result = {}
    for cfg in configs:
        j = cfg["joints"]
        result[cfg["name"]] = _angle_at_vertex(_xyz(df, j[0]), _xyz(df, j[1]), _xyz(df, j[2]))
    return pd.DataFrame(result, index=df.index)


def generate_engineered_features(yaml_path: str, raw_csv_path: str, output_dir: str | None = None) -> str:
    """Calculates geometric features directly from text-named landmark CSV tables."""
    _validate_inputs(yaml_path, raw_csv_path)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    feat_config = config.get("features_config", {}).get("engineered_features", {})
    if not feat_config or "joint_angles" not in feat_config:
        raise ValueError("Missing 'features_config.engineered_features.joint_angles' block in YAML.")

    df_raw = pd.read_csv(raw_csv_path)
    print(f"Loaded '{raw_csv_path}' -- {len(df_raw)} rows, {len(df_raw.columns)} columns.")

    # Validate that we have the necessary landmarks
    required_joints = _collect_required_indices(feat_config)
    _validate_landmark_columns(df_raw, required_joints)

    # Core Calculations Frame Generation: Extract only 'label' and skip any session/frame metadata
    feature_frames = []
    if "label" in df_raw.columns:
        feature_frames.append(df_raw[["label"]].copy())
        
    feature_frames.append(_compute_joint_angles(df_raw, feat_config["joint_angles"]))
    df_features = pd.concat(feature_frames, axis=1)
    
    # Save Output file logic matching your convention
    base = os.path.basename(raw_csv_path)
    out_filename = base.replace("_raw.csv", "_features.csv") if "_raw.csv" in base else f"{os.path.splitext(base)[0]}_features.csv"
    save_dir = os.path.abspath(output_dir) if output_dir else os.path.dirname(raw_csv_path)
    output_path = os.path.join(save_dir, out_filename)
    
    df_features.to_csv(output_path, index=False)
    print(f"Saved features CSV ({df_features.shape[0]} rows x {df_features.shape[1]} cols) -> {output_path}")
    return output_path


if __name__ == "__main__":
    generate_engineered_features(YAML_FILE, RAW_CSV, OUTPUT_DIR)