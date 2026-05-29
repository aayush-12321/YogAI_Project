# config.py — now just a loader, not a data store
import yaml
from pathlib import Path

def load_config(path: str = "../configs/models/models_config.yaml") -> dict:
    with open(Path(path)) as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# Keep these as module-level names so existing imports don't break
POSE_FEATURES = CONFIG["pose_features"]
HYPERPARAMETER_GRIDS = CONFIG["hyperparameter_grids"]
CV_CONFIG = CONFIG["cv_config"]
LABEL_COLUMN = CONFIG["columns"]["label"]
SOURCE_ID_COLUMN = CONFIG["columns"]["source_id"]
GROUP_COLUMNS = CONFIG["columns"]["group_columns"]
XGBOOST_RANDOM_SEARCH_ITER = CONFIG["xgboost_random_search_iter"]
FRAME_NUMBER_COLUMN = CONFIG["columns"]["frame_number"]