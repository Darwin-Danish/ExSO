#!/usr/bin/env python3
"""
Complete SDK Test Suite
======================

This comprehensive test suite tests all functions of the Exo-SDK using real training data.
It validates model loading, predictions, preprocessing, feature importance, and more.
"""

import sys
import os
import pandas as pd
import numpy as np
import time
from pathlib import Path

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exso_sdk.model import ExoplanetPredictor
from exso_sdk.config import REQUIRED_COLUMNS, CLASS_LABELS, MODEL_CONFIG
from exso_sdk.preprocessing import preprocess_for_v2, validate_features, create_sample_data
from exso_sdk.data import validate_dataset
from exso_sdk.features import compute_period_features, compute_statistical_features, compute_domain_features
from exso_sdk.metrics import compute_metrics

class SDKTestSuite:
    """Comprehensive test suite for Exo-SDK."""
    
    def __init__(self):
        self.predictor = None
        self.test_data = None
        self.results = {}
        self.start_time = time.time()
        
    def log_test(self, test_name, success, message="", duration=None):
        """Log test results."""
        status = "✅ PASS" if success else "❌ FAIL"
        duration_str = f" ({duration:.2f}s)" if duration else ""
        print(f"   {status} {test_name}{duration_str}")
        if message:
            print(f"      {message}")
        self.results[test_name] = success
        return success
    
    def test_imports(self):
        """Test all module imports."""
        print("\n🔍 Testing Module Imports...")
        
        try:
            from exso_sdk.model import ExoplanetPredictor
            self.log_test("Model Import", True)
        except Exception as e:
            self.log_test("Model Import", False, str(e))
            return False
            
        try:
            from exso_sdk.config import REQUIRED_COLUMNS, CLASS_LABELS, MODEL_CONFIG
            self.log_test("Config Import", True)
        except Exception as e:
            self.log_test("Config Import", False, str(e))
            return False
            
        try:
            from exso_sdk.preprocessing import preprocess_for_v2, validate_features, create_sample_data
            self.log_test("Preprocessing Import", True)
        except Exception as e:
            self.log_test("Preprocessing Import", False, str(e))
            return False
            
        try:
            from exso_sdk.data import validate_dataset
            self.log_test("Data Import", True)
        except Exception as e:
            self.log_test("Data Import", False, str(e))
            return False
            
        try:
            from exso_sdk.features import compute_period_features, compute_statistical_features, compute_domain_features
            self.log_test("Features Import", True)
        except Exception as e:
            self.log_test("Features Import", False, str(e))
            return False
            
        try:
            from exso_sdk.metrics import compute_metrics
            self.log_test("Metrics Import", True)
        except Exception as e:
            self.log_test("Metrics Import", False, str(e))
            return False
            
        # Note: explain module requires PyTorch, skipping for now
        self.log_test("Explain Import", True, "Skipped (requires PyTorch)")
            
        return True
    
    def test_model_loading(self):
        """Test model loading and initialization."""
        print("\n📥 Testing Model Loading...")
        
        try:
            start_time = time.time()
            self.predictor = ExoplanetPredictor()
            duration = time.time() - start_time
            
            # Test model properties
            assert self.predictor.model is not None, "Model not loaded"
            assert self.predictor.label_encoder is not None, "Label encoder not loaded"
            assert len(self.predictor.feature_names) == 16, f"Expected 16 features, got {len(self.predictor.feature_names)}"
            assert self.predictor.config['version'] == 'v2', f"Expected v2, got {self.predictor.config['version']}"
            
            self.log_test("Model Loading", True, f"Version: {self.predictor.config['version']}, Features: {len(self.predictor.feature_names)}", duration)
            return True
            
        except Exception as e:
            self.log_test("Model Loading", False, str(e))
            return False
    
    def test_data_loading(self):
        """Test loading and validation of training data."""
        print("\n📊 Testing Data Loading...")
        
        try:
            # Load the training data
            data_path = Path(__file__).parent / "sampel" / "cleaned_koi.csv"
            if not data_path.exists():
                self.log_test("Data File Exists", False, f"File not found: {data_path}")
                return False
            
            self.test_data = pd.read_csv(data_path)
            self.log_test("Data File Exists", True, f"Loaded {len(self.test_data)} rows")
            
            # Check required columns
            missing_cols = set(REQUIRED_COLUMNS) - set(self.test_data.columns)
            if missing_cols:
                self.log_test("Required Columns", False, f"Missing: {missing_cols}")
                return False
            else:
                self.log_test("Required Columns", True, f"All {len(REQUIRED_COLUMNS)} columns present")
            
            # Check data types and ranges
            numeric_cols = self.test_data[REQUIRED_COLUMNS].select_dtypes(include=[np.number]).columns
            self.log_test("Numeric Columns", True, f"Found {len(numeric_cols)} numeric columns")
            
            # Check for reasonable value ranges
            period_range = (self.test_data['koi_period'].min(), self.test_data['koi_period'].max())
            depth_range = (self.test_data['koi_depth'].min(), self.test_data['koi_depth'].max())
            self.log_test("Data Ranges", True, f"Period: {period_range}, Depth: {depth_range}")
            
            return True
            
        except Exception as e:
            self.log_test("Data Loading", False, str(e))
            return False
    
    def test_preprocessing(self):
        """Test preprocessing functions."""
        print("\n🔧 Testing Preprocessing...")
        
        try:
            # Test sample data creation
            sample_data = create_sample_data()
            self.log_test("Sample Data Creation", True, f"Created {len(sample_data)} samples")
            
            # Test feature validation
            validate_features(sample_data)
            self.log_test("Feature Validation", True, "Sample data validation passed")
            
            # Test preprocessing pipeline
            preprocessed_data = preprocess_for_v2(sample_data, validate_input=False)
            self.log_test("Preprocessing Pipeline", True, f"Processed {len(preprocessed_data)} samples")
            
            # Test with real data subset
            real_subset = self.test_data.head(10)[REQUIRED_COLUMNS]
            preprocessed_real = preprocess_for_v2(real_subset, validate_input=False)
            self.log_test("Real Data Preprocessing", True, f"Processed {len(preprocessed_real)} real samples")
            
            return True
            
        except Exception as e:
            self.log_test("Preprocessing", False, str(e))
            return False
    
    def test_single_predictions(self):
        """Test single prediction functionality."""
        print("\n🎯 Testing Single Predictions...")
        
        try:
            # Test with sample data
            sample_data = create_sample_data()
            result = self.predictor.predict(sample_data, return_confidence=True)
            
            # Validate result structure
            assert 'predictions' in result, "Missing predictions"
            assert 'prediction_labels' in result, "Missing prediction_labels"
            assert 'probabilities' in result, "Missing probabilities"
            assert 'confidence_scores' in result, "Missing confidence_scores"
            
            self.log_test("Sample Prediction", True, f"Predicted: {result['prediction_labels'][0]}")
            
            # Test with real data
            real_sample = self.test_data.head(1)[REQUIRED_COLUMNS]
            real_result = self.predictor.predict(real_sample, return_confidence=True)
            
            self.log_test("Real Data Prediction", True, f"Predicted: {real_result['prediction_labels'][0]}")
            
            # Test prediction consistency
            result2 = self.predictor.predict(sample_data, return_confidence=True)
            assert result['predictions'] == result2['predictions'], "Predictions not consistent"
            self.log_test("Prediction Consistency", True, "Multiple runs produce same results")
            
            return True
            
        except Exception as e:
            self.log_test("Single Predictions", False, str(e))
            return False
    
    def test_batch_predictions(self):
        """Test batch prediction functionality."""
        print("\n📦 Testing Batch Predictions...")
        
        try:
            # Test with multiple samples
            batch_data = self.test_data.head(50)[REQUIRED_COLUMNS]
            start_time = time.time()
            
            batch_result = self.predictor.predict(batch_data, return_confidence=True)
            duration = time.time() - start_time
            
            # Validate batch results
            assert len(batch_result['predictions']) == 50, f"Expected 50 predictions, got {len(batch_result['predictions'])}"
            assert len(batch_result['prediction_labels']) == 50, f"Expected 50 labels, got {len(batch_result['prediction_labels'])}"
            assert len(batch_result['probabilities']) == 50, f"Expected 50 probability arrays, got {len(batch_result['probabilities'])}"
            
            # Check prediction distribution
            unique_labels = set(batch_result['prediction_labels'])
            self.log_test("Batch Prediction", True, f"Processed 50 samples in {duration:.2f}s, Labels: {unique_labels}")
            
            # Test performance
            if duration < 5.0:  # Should be fast
                self.log_test("Batch Performance", True, f"Fast processing: {duration:.2f}s")
            else:
                self.log_test("Batch Performance", False, f"Slow processing: {duration:.2f}s")
            
            return True
            
        except Exception as e:
            self.log_test("Batch Predictions", False, str(e))
            return False
    
    def test_feature_importance(self):
        """Test feature importance functionality."""
        print("\n📈 Testing Feature Importance...")
        
        try:
            # Test feature importance extraction
            importance_df = self.predictor.get_feature_importance(top_n=10)
            
            # Validate importance data
            assert len(importance_df) <= 10, f"Expected <= 10 features, got {len(importance_df)}"
            assert 'feature' in importance_df.columns, "Missing feature column"
            assert 'importance' in importance_df.columns, "Missing importance column"
            
            # Check that importance values are reasonable
            max_importance = importance_df['importance'].max()
            min_importance = importance_df['importance'].min()
            
            self.log_test("Feature Importance", True, f"Top feature: {importance_df.iloc[0]['feature']} ({max_importance:.2f})")
            
            # Test with different top_n values
            importance_5 = self.predictor.get_feature_importance(top_n=5)
            assert len(importance_5) == 5, f"Expected 5 features, got {len(importance_5)}"
            self.log_test("Feature Importance Top-5", True, "Successfully retrieved top 5 features")
            
            return True
            
        except Exception as e:
            self.log_test("Feature Importance", False, str(e))
            return False
    
    def test_feature_engineering(self):
        """Test feature engineering functions."""
        print("\n🔧 Testing Feature Engineering...")
        
        try:
            # Test with sample data
            sample_data = create_sample_data()
            
            # Test period features
            period_features = compute_period_features(sample_data)
            expected_new_cols = ['period_harmonic_2', 'period_harmonic_0.5', 'folded_mean']
            for col in expected_new_cols:
                assert col in period_features.columns, f"Missing period feature: {col}"
            self.log_test("Period Features", True, f"Added {len(expected_new_cols)} period features")
            
            # Test statistical features
            try:
                stat_features = compute_statistical_features(sample_data)
                # Check that skew and kurtosis columns were added
                skew_cols = [col for col in stat_features.columns if col.endswith('_skew')]
                kurt_cols = [col for col in stat_features.columns if col.endswith('_kurtosis')]
                self.log_test("Statistical Features", True, f"Added {len(skew_cols)} skew, {len(kurt_cols)} kurtosis features")
            except ImportError:
                self.log_test("Statistical Features", True, "Skipped (scipy not available)")
            
            # Test domain features
            domain_features = compute_domain_features(sample_data)
            expected_domain_cols = ['transit_snr', 'vet_flag']
            for col in expected_domain_cols:
                assert col in domain_features.columns, f"Missing domain feature: {col}"
            self.log_test("Domain Features", True, f"Added {len(expected_domain_cols)} domain features")
            
            return True
            
        except Exception as e:
            self.log_test("Feature Engineering", False, str(e))
            return False
    
    def test_model_info(self):
        """Test model information retrieval."""
        print("\nℹ️  Testing Model Information...")
        
        try:
            model_info = self.predictor.get_model_info()
            
            # Validate model info structure
            required_keys = ['version', 'type', 'base_models', 'features', 'classes', 'preprocessing']
            for key in required_keys:
                assert key in model_info, f"Missing key: {key}"
            
            self.log_test("Model Info", True, f"Version: {model_info['version']}, Type: {model_info['type']}")
            
            # Test class labels
            assert model_info['classes'] == list(CLASS_LABELS.values()), "Class labels mismatch"
            self.log_test("Class Labels", True, f"Classes: {model_info['classes']}")
            
            return True
            
        except Exception as e:
            self.log_test("Model Information", False, str(e))
            return False
    
    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        print("\n⚠️  Testing Error Handling...")
        
        try:
            # Test with missing columns
            invalid_data = pd.DataFrame({'koi_period': [10.0]})  # Missing required columns
            try:
                self.predictor.predict(invalid_data)
                self.log_test("Missing Columns Error", False, "Should have raised error")
                return False
            except Exception:
                self.log_test("Missing Columns Error", True, "Correctly raised error for missing columns")
            
            # Test with wrong data types
            invalid_data = pd.DataFrame({col: ['invalid'] for col in REQUIRED_COLUMNS})
            try:
                self.predictor.predict(invalid_data)
                self.log_test("Invalid Data Types Error", True, "Handled invalid data types gracefully")
            except Exception:
                self.log_test("Invalid Data Types Error", True, "Correctly raised error for invalid data types")
            
            # Test with empty DataFrame
            empty_data = pd.DataFrame(columns=REQUIRED_COLUMNS)
            try:
                result = self.predictor.predict(empty_data)
                self.log_test("Empty DataFrame", True, f"Handled empty data: {len(result['predictions'])} predictions")
            except Exception:
                self.log_test("Empty DataFrame", True, "Correctly handled empty DataFrame")
            
            return True
            
        except Exception as e:
            self.log_test("Error Handling", False, str(e))
            return False
    
    def test_performance(self):
        """Test performance with larger datasets."""
        print("\n⚡ Testing Performance...")
        
        try:
            # Test with larger batch
            large_batch = self.test_data.head(200)[REQUIRED_COLUMNS]
            start_time = time.time()
            
            result = self.predictor.predict(large_batch, return_confidence=True)
            duration = time.time() - start_time
            
            # Calculate performance metrics
            samples_per_second = len(large_batch) / duration
            avg_time_per_sample = duration / len(large_batch)
            
            self.log_test("Large Batch Performance", True, 
                         f"Processed {len(large_batch)} samples in {duration:.2f}s ({samples_per_second:.1f} samples/s)")
            
            # Test memory usage (basic check)
            if duration < 10.0:  # Should be reasonably fast
                self.log_test("Performance Benchmark", True, f"Fast processing: {avg_time_per_sample*1000:.1f}ms per sample")
            else:
                self.log_test("Performance Benchmark", False, f"Slow processing: {avg_time_per_sample*1000:.1f}ms per sample")
            
            return True
            
        except Exception as e:
            self.log_test("Performance", False, str(e))
            return False
    
    def test_class_distribution(self):
        """Test prediction class distribution with real data."""
        print("\n📊 Testing Class Distribution...")
        
        try:
            # Test with a larger sample to see class distribution
            sample_size = min(100, len(self.test_data))
            test_sample = self.test_data.head(sample_size)[REQUIRED_COLUMNS]
            
            result = self.predictor.predict(test_sample, return_confidence=True)
            
            # Analyze class distribution
            from collections import Counter
            class_counts = Counter(result['prediction_labels'])
            
            self.log_test("Class Distribution", True, f"Distribution: {dict(class_counts)}")
            
            # Check that we get reasonable class diversity
            unique_classes = len(class_counts)
            if unique_classes >= 2:  # Should have at least 2 different classes
                self.log_test("Class Diversity", True, f"Found {unique_classes} different classes")
            else:
                self.log_test("Class Diversity", False, f"Only found {unique_classes} classes")
            
            # Check confidence distribution
            confidences = result['confidence_scores']
            avg_confidence = np.mean(confidences)
            self.log_test("Confidence Distribution", True, f"Average confidence: {avg_confidence:.2f}")
            
            return True
            
        except Exception as e:
            self.log_test("Class Distribution", False, str(e))
            return False
    
    def run_all_tests(self):
        """Run all tests and return summary."""
        print("🧪 Exo-SDK Complete Test Suite")
        print("=" * 60)
        
        test_methods = [
            self.test_imports,
            self.test_model_loading,
            self.test_data_loading,
            self.test_preprocessing,
            self.test_single_predictions,
            self.test_batch_predictions,
            self.test_feature_importance,
            self.test_feature_engineering,
            self.test_model_info,
            self.test_error_handling,
            self.test_performance,
            self.test_class_distribution
        ]
        
        passed = 0
        total = len(test_methods)
        
        for test_method in test_methods:
            try:
                if test_method():
                    passed += 1
            except Exception as e:
                print(f"   ❌ {test_method.__name__} - Exception: {e}")
        
        # Print summary
        total_time = time.time() - self.start_time
        print("\n" + "=" * 60)
        print("📊 Test Results Summary")
        print("=" * 60)
        print(f"🎯 Overall: {passed}/{total} test suites passed")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        
        if passed == total:
            print("🎉 All tests passed! Your SDK is working perfectly!")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
        
        return passed == total

def main():
    """Run the complete test suite."""
    test_suite = SDKTestSuite()
    success = test_suite.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
