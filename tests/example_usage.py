#!/usr/bin/env python3
"""
Exo-SDK Example Usage
=====================

Super simple examples of how to use the Exo-SDK.

After installing: pip install exso-sdk
"""

# Example 1: Direct prediction using ExoplanetPredictor
print("🚀 Example 1: Using ExoplanetPredictor Class")
print("=" * 50)

try:
    import exso_sdk
    import pandas as pd
    
    # Initialize predictor
    predictor = exso_sdk.ExoplanetPredictor()
    print("✅ Predictor initialized successfully!")
    
except ImportError:
    print("❌ Exo-SDK not installed!")
    print("💡 Install it with: pip install exso-sdk")
except Exception as e:
    print(f"❌ Error initializing predictor: {e}")

# Example 2: Quick prediction function
print("\n🎯 Example 2: Quick Prediction Function")
print("=" * 40)

try:
    import exso_sdk
    import pandas as pd
    
    # Create sample data
    sample_data = {
        'koi_period': 10.5,
        'koi_depth': 1500.0,
        'koi_prad': 2.0,
        'koi_sma': 0.08,
        'koi_teq': 900.0,
        'koi_insol': 150.0,
        'koi_model_snr': 50.0,
        'koi_time0bk': 2454833.0,
        'koi_duration': 0.3,
        'koi_incl': 89.5,
        'koi_srho': 1.4,
        'koi_srad': 1.0,
        'koi_smass': 1.1,
        'koi_steff': 6000.0,
        'koi_slogg': 4.4,
        'koi_smet': 0.0
    }
    
    # Make prediction using quick function
    df = pd.DataFrame([sample_data])
    result = exso_sdk.predict(df)
    
    print(f"Prediction: {result['prediction_labels'][0]}")
    print(f"Confidence: {result['confidence_scores'][0]:.2f}")
    print(f"Probabilities: {result['probabilities'][0]}")
    
except ImportError:
    print("❌ Exo-SDK not installed!")
    print("💡 Install it with: pip install exso-sdk")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n✅ Examples completed!")
print("💡 For more examples, check the documentation at:")
print("   https://github.com/yourname/exso-sdk")
