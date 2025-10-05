#!/usr/bin/env python3
"""
Demo: Using Exo-SDK from PyPI
=============================

This script demonstrates how to use the published exso-sdk package from PyPI.
"""

import sys

def check_package():
    """Check if the package is installed."""
    try:
        import exso_sdk
        print(f"✅ exso-sdk package found!")
        
        # Try to get version
        try:
            version = exso_sdk.__version__
            print(f"📦 Version: {version}")
        except AttributeError:
            print("📦 Version: unknown")
        
        return True
    except ImportError:
        print("❌ exso-sdk package not found!")
        print("💡 Install it with:")
        print("   pip install -i https://test.pypi.org/simple/ exso-sdk")
        print("   or")
        print("   pip install exso-sdk")
        return False

def demo_prediction():
    """Demonstrate prediction using the PyPI package."""
    try:
        from exso_sdk.model import ExoplanetPredictor, predict_exoplanet
        from exso_sdk.config import REQUIRED_COLUMNS, CLASS_LABELS
        import pandas as pd
        
        print("\n🚀 Exo-SDK PyPI Package Demo")
        print("=" * 50)
        
        # Initialize predictor
        print("📥 Loading model...")
        predictor = ExoplanetPredictor()
        print("✅ Model loaded successfully!")
        
        # Get model info
        model_info = predictor.get_model_info()
        print(f"📊 Model info: {model_info}")
        
        # Create test data
        test_data = {
            "koi_period": 10.5,
            "koi_depth": 1500.0,
            "koi_prad": 2.0,
            "koi_sma": 0.08,
            "koi_teq": 900.0,
            "koi_insol": 150.0,
            "koi_model_snr": 50.0,
            "koi_time0bk": 2454833.0,
            "koi_duration": 0.3,
            "koi_incl": 89.5,
            "koi_srho": 1.4,
            "koi_srad": 1.0,
            "koi_smass": 1.1,
            "koi_steff": 6000.0,
            "koi_slogg": 4.4,
            "koi_smet": 0.0
        }
        
        print(f"\n🎯 Testing with sample data:")
        print(f"   Period: {test_data['koi_period']} days")
        print(f"   Depth: {test_data['koi_depth']} ppm")
        print(f"   Radius: {test_data['koi_prad']} Earth radii")
        print(f"   SNR: {test_data['koi_model_snr']}")
        
        # Make prediction
        df = pd.DataFrame([test_data])
        results = predictor.predict(df, return_confidence=True)
        
        prediction = results['predictions'][0]
        prediction_label = results['prediction_labels'][0]
        confidence = results['confidence_scores'][0]
        probabilities = results['probabilities'][0]
        
        print(f"\n🎯 Prediction Results:")
        print(f"   Prediction: {prediction} ({prediction_label})")
        print(f"   Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
        
        print(f"\n📈 Class Probabilities:")
        for i, (class_id, class_name) in enumerate(CLASS_LABELS.items()):
            prob = probabilities[i]
            print(f"      {class_name}: {prob:.3f} ({prob*100:.1f}%)")
        
        # Feature importance
        print(f"\n📊 Feature Importance (Top 5):")
        importance_df = predictor.get_feature_importance(top_n=5)
        for _, row in importance_df.iterrows():
            print(f"      {row['feature']}: {row['importance']:.3f}")
        
        # Quick prediction function
        print(f"\n⚡ Quick Prediction Test:")
        quick_results = predict_exoplanet(df)
        print(f"   Quick prediction: {quick_results['prediction_labels'][0]}")
        print(f"   Confidence: {quick_results['confidence_scores'][0]:.3f}")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"✅ The exso-sdk PyPI package is working perfectly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_additional_features():
    """Demonstrate additional SDK features."""
    try:
        from exso_sdk import get_model_info, get_feature_importance
        
        print(f"\n🔧 Additional SDK Features")
        print("=" * 40)
        
        # Get model info
        model_info = get_model_info()
        print(f"📊 Model info: {model_info}")
        
        # Get feature importance
        feature_importance = get_feature_importance(top_n=5)
        print(f"📈 Top 5 feature importance:")
        for _, row in feature_importance.iterrows():
            print(f"      {row['feature']}: {row['importance']:.3f}")
        
        print("✅ Additional features demo completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Additional features demo failed: {e}")
        return False

def main():
    """Main demo function."""
    print("🧪 Exo-SDK PyPI Package Demo")
    print("=" * 50)
    
    # Check if package is installed
    if not check_package():
        return
    
    # Demo prediction functionality
    if not demo_prediction():
        return
    
    # Demo additional features
    demo_additional_features()
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Run tests: python run_tests.py")
    print(f"   2. Check documentation: https://github.com/yourname/exso-sdk")
    print(f"   3. Try the quick_start.py example")

if __name__ == "__main__":
    main()
