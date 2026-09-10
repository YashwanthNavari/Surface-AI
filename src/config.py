"""
Configuration Loader and Path Manager
Loads config.yaml and provides paths and settings as structured objects.
"""
from pathlib import Path
import yaml

# Resolve project root relative to this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "config.yaml"

def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# Convenient Path Constants
RAW_DATA_DIR = PROJECT_ROOT / CONFIG["paths"]["raw_data_dir"]
PROCESSED_DATA_DIR = PROJECT_ROOT / CONFIG["paths"]["processed_data_dir"]
SPLITS_DIR = PROJECT_ROOT / CONFIG["paths"]["splits_dir"]
MODELS_DIR = PROJECT_ROOT / CONFIG["paths"]["models_dir"]
RESULTS_DIR = PROJECT_ROOT / CONFIG["paths"]["results_dir"]
METRICS_DIR = PROJECT_ROOT / CONFIG["paths"]["metrics_dir"]
CONFUSION_MATRICES_DIR = PROJECT_ROOT / CONFIG["paths"]["confusion_matrices_dir"]
TRAINING_CURVES_DIR = PROJECT_ROOT / CONFIG["paths"]["training_curves_dir"]
GRADCAM_DIR = PROJECT_ROOT / CONFIG["paths"]["gradcam_dir"]

# Ensure directories exist
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SPLITS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    METRICS_DIR,
    CONFUSION_MATRICES_DIR,
    TRAINING_CURVES_DIR,
    GRADCAM_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset specifications
CLASSES = CONFIG["dataset"]["classes"]
NUM_CLASSES = CONFIG["dataset"]["num_classes"]
IMAGE_SIZE = tuple(CONFIG["dataset"]["image_size"])
CHANNELS = CONFIG["dataset"]["channels"]
RANDOM_SEED = CONFIG["dataset"]["random_seed"]

# Class mapping dictionaries
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}
