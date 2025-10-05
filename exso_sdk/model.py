"""
Exoplanet Candidate Classification Model - V2 Stacking Pipeline
==============================================================

This module provides the interface for the V2 exoplanet classification model,
which uses a stacking ensemble of LightGBM, XGBoost, and CatBoost classifiers
with robust preprocessing and NaN handling.
"""

import os
import joblib
import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Union, Optional

from .config import load_model, load_label_encoder, REQUIRED_COLUMNS, CLASS_LABELS, MODEL_CONFIG

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class ExoplanetPredictor:
    """
    V2 Exoplanet Candidate Classifier using Stacking Ensemble.
    
    This class provides a clean interface to the V2 model which includes:
    - Robust preprocessing with different strategies for critical vs auxiliary features
    - Stacking ensemble of LightGBM, XGBoost, and CatBoost
    - Automatic NaN handling
    - Confidence scoring
    """
    
    def __init__(self, model_path: Optional[str] = None, label_encoder_path: Optional[str] = None):
        """
        Initialize the ExoplanetPredictor with V2 model.
        
        Args:
            model_path: Optional path to model file. If None, uses default from config.
            label_encoder_path: Optional path to label encoder. If None, uses default from config.
        """
        self.model = None
        self.label_encoder = None
        self.feature_names = REQUIRED_COLUMNS
        self.class_labels = CLASS_LABELS
        self.config = MODEL_CONFIG
        
        # Load model and label encoder
        self._load_model(model_path, label_encoder_path)
    
    def _load_model(self, model_path: Optional[str] = None, label_encoder_path: Optional[str] = None):
        """Load the V2 model and label encoder."""
        try:
            if model_path:
                self.model = joblib.load(model_path)
            else:
                self.model = load_model()
            
            if label_encoder_path:
                self.label_encoder = joblib.load(label_encoder_path)
            else:
                self.label_encoder = load_label_encoder()
                
            print(f"✓ Loaded V2 model successfully")
            print(f"  - Model type: {type(self.model)}")
            print(f"  - Features: {len(self.feature_names)}")
            print(f"  - Classes: {list(self.class_labels.values())}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load V2 model: {e}")
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray, List], return_confidence: bool = True) -> Dict:
        """
        Predict exoplanet candidates using the V2 stacking model.
        
        Args:
            data: Input data as DataFrame, numpy array, or list
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with predictions and optional confidence scores
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Initialize ExoplanetPredictor first.")
        
        # Convert input to DataFrame if needed
        if isinstance(data, (list, np.ndarray)):
            if isinstance(data, list):
                data = np.array(data)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            data = pd.DataFrame(data, columns=self.feature_names)
        elif isinstance(data, pd.DataFrame):
            # Validate required columns
            missing_cols = set(self.feature_names) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            data = data[self.feature_names]  # Ensure correct column order
        
        # Make predictions
        try:
            predictions = self.model.predict(data)
            probabilities = self.model.predict_proba(data)
            
            # Decode predictions using label encoder if available
            if self.label_encoder is not None:
                decoded_predictions = self.label_encoder.inverse_transform(predictions)
                prediction_labels = [self.class_labels[pred] for pred in decoded_predictions]
            else:
                # Fallback to direct mapping
                prediction_labels = [self.class_labels[pred] for pred in predictions]
            
            result = {
                'predictions': decoded_predictions.tolist() if self.label_encoder is not None else predictions.tolist(),
                'prediction_labels': prediction_labels,
                'probabilities': probabilities.tolist()
            }
            
            if return_confidence:
                # Calculate confidence as max probability
                confidence_scores = np.max(probabilities, axis=1)
                result['confidence_scores'] = confidence_scores.tolist()
                result['avg_confidence'] = float(np.mean(confidence_scores))
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def predict_single(self, features: Dict[str, float]) -> Dict:
        """
        Predict a single exoplanet candidate.
        
        Args:
            features: Dictionary with feature names as keys and values
            
        Returns:
            Dictionary with prediction results
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])
        return self.predict(df)
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from the first base model (LightGBM).
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        
        try:
            # Access the stacking classifier
            stacking_classifier = self.model.named_steps['classifier']
            
            # Get the first base model (LightGBM)
            first_base_model = stacking_classifier.estimators_[0]
            
            if hasattr(first_base_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': first_base_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                return importance_df.head(top_n)
            else:
                raise AttributeError("Feature importance not available for base model")
                
        except Exception as e:
            raise RuntimeError(f"Could not extract feature importance: {e}")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'version': self.config['version'],
            'type': self.config['type'],
            'base_models': self.config['base_models'],
            'features': len(self.feature_names),
            'classes': list(self.class_labels.values()),
            'preprocessing': self.config['preprocessing']
        }


# Convenience functions for backward compatibility
def load_exoplanet_model(model_path: Optional[str] = None) -> ExoplanetPredictor:
    """Load the V2 exoplanet model."""
    return ExoplanetPredictor(model_path)


def predict_exoplanet(data: Union[pd.DataFrame, np.ndarray, List], 
                     model_path: Optional[str] = None) -> Dict:
    """
    Quick prediction function using V2 model.
    
    Args:
        data: Input data
        model_path: Optional model path
        
    Returns:
        Prediction results
    """
    predictor = ExoplanetPredictor(model_path)
    return predictor.predict(data)


# Legacy function names for backward compatibility
def predict(model, sample):
    """Legacy prediction function - now uses V2 model."""
    if isinstance(model, ExoplanetPredictor):
        return model.predict_single(sample)
    else:
        # For backward compatibility with old model objects
        predictor = ExoplanetPredictor()
        result = predictor.predict_single(sample)
        pred_class = result['predictions'][0]
        probs = np.array(result['probabilities'][0])
        return pred_class, probs


def load_model(input_dim=None, config=None, path=None):
    """Legacy model loading function - now returns V2 predictor."""
    # Avoid circular import by using the config module directly
    import joblib
    if path:
        model = joblib.load(path)
    else:
        from .config import load_model as config_load_model
        model = config_load_model()
    return model


def save_model(model, path=None):
    """Legacy save function - not applicable for V2 model."""
    raise NotImplementedError("V2 model saving not supported. Use the training script instead.")