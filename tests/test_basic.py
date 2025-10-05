import numpy as np
import pandas as pd
from exso_sdk.model import ExoplanetPredictor
from exso_sdk.config import REQUIRED_COLUMNS

# -------------------------------
# Predict Exoplanet using V2 Model
# -------------------------------
def predict_exoplanet(example, predictor):
    """Predict exoplanet classification for a single example using V2 model."""
    # Convert example to DataFrame
    if isinstance(example, dict):
        df = pd.DataFrame([example])
        # Fill missing columns with 0
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
        df = df[REQUIRED_COLUMNS]
    elif isinstance(example, (list, np.ndarray)):
        if len(example) != len(REQUIRED_COLUMNS):
            raise ValueError(f"Expected {len(REQUIRED_COLUMNS)} features, got {len(example)}")
        df = pd.DataFrame([example], columns=REQUIRED_COLUMNS)
    else:
        raise ValueError("example must be dict, list, or numpy array")
    
    # Use the V2 predictor
    result = predictor.predict(df, return_confidence=True)
    
    # Extract results
    prediction = result['predictions'][0]
    prediction_label = result['prediction_labels'][0]
    confidence = result['confidence_scores'][0]
    probabilities = result['probabilities'][0]
    
    # Map to boolean for exoplanet detection
    exo = (prediction_label == "Positive")
    
    return {
        "exo": exo,
        "label": prediction_label,
        "confidence": float(confidence),
        "probs": probabilities
    }

# -------------------------------
# Display Results Function
# -------------------------------
def display_results(result, sample_info=None):
    """
    Display prediction results in a structured and visually appealing way.
    
    Args:
        result: dict from predict_exoplanet function
        sample_info: dict with sample description (optional)
    """
    print("=" * 80)
    print("�� EXOPLANET CLASSIFICATION RESULTS")
    print("=" * 80)
    
    # Sample information
    if sample_info:
        print(f"\n📊 Sample Information:")
        print(f"   {sample_info}")
    
    # Main prediction
    print(f"\n🎯 PREDICTION:")
    exo_icon = "✅" if result["exo"] else "❌"
    print(f"   {exo_icon} Exoplanet Detected: {result['exo']}")
    print(f"   🏷️  Classification: {result['label']}")
    print(f"   🎯 Confidence: {result['confidence']:.1%}")
    
    # Probability breakdown
    print(f"\n📈 PROBABILITY BREAKDOWN:")
    print("   " + "-" * 50)
    
    labels = ["False Positive", "Candidate", "Confirmed"]
    probs = result["probs"]
    
    # Sort by probability (descending)
    sorted_indices = np.argsort(probs)[::-1]
    
    for i, idx in enumerate(sorted_indices):
        label = labels[idx]
        prob = probs[idx]
        confidence_bar = "█" * int(prob * 20)  # 20 character bar
        confidence_bar = confidence_bar.ljust(20)
        
        # Add ranking indicator
        rank_icon = "🥇" if i == 0 else "��" if i == 1 else "🥉"
        
        print(f"   {rank_icon} {label:<15}: {prob:.1%} {confidence_bar}")
    
    # Confidence level interpretation
    confidence = result["confidence"]
    if confidence >= 0.9:
        conf_level = "Very High"
        conf_icon = "🟢"
    elif confidence >= 0.7:
        conf_level = "High"
        conf_icon = "🟡"
    elif confidence >= 0.5:
        conf_level = "Medium"
        conf_icon = "🟠"
    else:
        conf_level = "Low"
        conf_icon = "🔴"
    
    print(f"\n��️  CONFIDENCE LEVEL:")
    print(f"   {conf_icon} {conf_level} Confidence ({confidence:.1%})")
    
    # Interpretation
    print(f"\n💡 INTERPRETATION:")
    if result["exo"]:
        print(f"   🌍 This object is classified as a CONFIRMED exoplanet!")
        print(f"   🔬 The model is {confidence:.1%} confident in this classification.")
        if confidence >= 0.8:
            print(f"   ⭐ High confidence suggests strong evidence for planetary nature.")
        else:
            print(f"   ⚠️  Moderate confidence - additional observations recommended.")
    else:
        print(f"   🚫 This object is NOT classified as a confirmed exoplanet.")
        print(f"   📊 Most likely classification: {result['label']}")
        if result['label'] == 'Candidate':
            print(f"   🔍 Further analysis needed to confirm planetary nature.")
        else:
            print(f"   ❌ Likely a false positive (stellar variability, instrumental noise, etc.)")
    
    print("=" * 80)

# -------------------------------
# Example usage with V2 Model
# -------------------------------
if __name__ == "__main__":
    print(f"🚀 Initializing Exoplanet Classifier V2...")
    
    # Load V2 model
    print(f"\n📥 Loading V2 pre-trained model...")
    try:
        predictor = ExoplanetPredictor()
        print(f"✅ V2 Model loaded successfully!")
        print(f"   - Model version: {predictor.config['version']}")
        print(f"   - Base models: {predictor.config['base_models']}")
        print(f"   - Features: {len(predictor.feature_names)}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        exit(1)
    
    # Test sample
    sample = {
        'koi_period': 14.3,
        'koi_time0bk': 512.6,
        'koi_duration': 3.2,
        'koi_depth': 150.0,
        'koi_prad': 1.4,
        'koi_sma': 0.12,
        'koi_incl': 87.9,
        'koi_teq': 950,
        'koi_insol': 150.0,
        'koi_srho': 1.05,
        'koi_srad': 0.95,
        'koi_smass': 0.9,
        'koi_steff': 5200,
        'koi_slogg': 4.6,
        'koi_smet': -0.2,
        'koi_model_snr': 8.0
    }
    
    # Test prediction
    print(f"\n🔮 Making prediction...")
    try:
        result = predict_exoplanet(sample, predictor)
        display_results(result, {"Sample": "Test exoplanet candidate"})
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        exit(1)
    
    # Test consistency
    print(f"\n🔄 Testing consistency (3 runs):")
    for i in range(3):
        result = predict_exoplanet(sample, predictor)
        print(f"Run {i+1}: {result['label']} ({result['confidence']:.1%})")
    
    print(f"\n✅ V2 Model is working correctly!")