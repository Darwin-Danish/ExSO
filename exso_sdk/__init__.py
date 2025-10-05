"""
Exo-SDK Package
===============

Exoplanet Candidate Classification SDK.

Quick Start:
    import exso_sdk
    predictor = exso_sdk.ExoplanetPredictor()
    results = predictor.predict(data)
"""

from .model import ExoplanetPredictor, predict_exoplanet
from .config import REQUIRED_COLUMNS, CLASS_LABELS, MODEL_CONFIG

# Version
__version__ = "2.1.11"

def predict(data, return_confidence=True):
    """
    Quick prediction function.
    
    Args:
        data: Input data (DataFrame, dict, or list)
        return_confidence: Whether to return confidence scores
    
    Returns:
        Prediction results
    """
    predictor = ExoplanetPredictor()
    return predictor.predict(data, return_confidence=return_confidence)

def get_model_info():
    """Get model information."""
    predictor = ExoplanetPredictor()
    return predictor.get_model_info()

def get_feature_importance(top_n=10):
    """Get feature importance."""
    predictor = ExoplanetPredictor()
    return predictor.get_feature_importance(top_n=top_n)

# Export main classes and functions
__all__ = [
    'predict', 'get_model_info', 'get_feature_importance',
    'ExoplanetPredictor', 'predict_exoplanet',
    'REQUIRED_COLUMNS', 'CLASS_LABELS', 'MODEL_CONFIG'
]