import os
import joblib
from importlib import resources


def get_model_path():
    """Return path to v2 stacking pipeline model."""
    env_path = os.environ.get("EXSO_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    
    # First try package directory (for installed package)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_model_path = os.path.join(current_dir, "model_data", "v2", "exo_stacking_pipeline.pkl")
    if os.path.exists(package_model_path):
        return package_model_path
    
    # Fallback to development directory
    dev_model_path = os.path.join(current_dir, "..", "model", "v2", "exo_stacking_pipeline.pkl")
    if os.path.exists(dev_model_path):
        return dev_model_path
    
    raise FileNotFoundError(f"V2 model not found. Tried: {package_model_path}, {dev_model_path}")


def get_label_encoder_path():
    """Return path to v2 label encoder."""
    env_path = os.environ.get("EXSO_LABEL_ENCODER_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    
    # First try package directory (for installed package)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_encoder_path = os.path.join(current_dir, "model_data", "v2", "exo_label_encoder.pkl")
    if os.path.exists(package_encoder_path):
        return package_encoder_path
    
    # Fallback to development directory
    dev_encoder_path = os.path.join(current_dir, "..", "model", "v2", "exo_label_encoder.pkl")
    if os.path.exists(dev_encoder_path):
        return dev_encoder_path
    
    raise FileNotFoundError(f"V2 label encoder not found. Tried: {package_encoder_path}, {dev_encoder_path}")


def load_model():
    """Load the v2 stacking pipeline model."""
    model_path = get_model_path()
    return joblib.load(model_path)


def load_label_encoder():
    """Load the v2 label encoder."""
    encoder_path = get_label_encoder_path()
    return joblib.load(encoder_path)


# V2 Model Feature Definitions (from the enhanced model)
CRITICAL_FEATURES = [
    'koi_period', 'koi_depth', 'koi_prad', 'koi_sma', 'koi_teq',
    'koi_insol', 'koi_model_snr'  # Most important features based on research
]

AUXILIARY_FEATURES = [
    'koi_time0bk', 'koi_duration', 'koi_incl', 'koi_srho',
    'koi_srad', 'koi_smass', 'koi_steff', 'koi_slogg', 'koi_smet'
]

# All features required for the v2 model
REQUIRED_COLUMNS = CRITICAL_FEATURES + AUXILIARY_FEATURES

# Class labels for v2 model (from label encoder analysis)
CLASS_LABELS = {
    -1: 'False Positive',
    0: 'Candidate', 
    1: 'Positive'
}

# Model configuration
MODEL_CONFIG = {
    'version': 'v2',
    'type': 'stacking_classifier',
    'base_models': ['lightgbm', 'xgboost', 'catboost'],
    'preprocessing': 'column_transformer_with_nan_handling',
    'feature_importance': True
}