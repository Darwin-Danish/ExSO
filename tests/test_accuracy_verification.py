#!/usr/bin/env python3
"""
Accuracy Verification Test
=========================

This test loads the complete training dataset, makes predictions on all samples,
and compares them against the ground truth labels to verify model accuracy.
"""

import sys
import os
import pandas as pd
import numpy as np
import time
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exso_sdk.model import ExoplanetPredictor
from exso_sdk.config import REQUIRED_COLUMNS, CLASS_LABELS

class AccuracyVerification:
    """Comprehensive accuracy verification using the complete training dataset."""
    
    def __init__(self):
        self.predictor = None
        self.training_data = None
        self.ground_truth = None
        self.predictions = None
        self.start_time = time.time()
        
    def load_data(self):
        """Load the complete training dataset."""
        print("📊 Loading Complete Training Dataset...")
        
        data_path = Path(__file__).parent / "sampel" / "cleaned_koi.csv"
        if not data_path.exists():
            print(f"❌ Data file not found: {data_path}")
            return False
        
        # Load the data
        self.training_data = pd.read_csv(data_path)
        print(f"✅ Loaded {len(self.training_data)} samples")
        
        # Extract ground truth labels
        if 'koi_disposition' in self.training_data.columns:
            self.ground_truth = self.training_data['koi_disposition'].values
            print(f"✅ Ground truth labels: {Counter(self.ground_truth)}")
        else:
            print("❌ Ground truth column 'koi_disposition' not found")
            return False
        
        # Prepare features
        self.features = self.training_data[REQUIRED_COLUMNS]
        print(f"✅ Features prepared: {self.features.shape}")
        
        return True
    
    def load_model(self):
        """Load the V2 model."""
        print("\n🤖 Loading V2 Model...")
        
        try:
            self.predictor = ExoplanetPredictor()
            print(f"✅ Model loaded successfully")
            print(f"   - Version: {self.predictor.config['version']}")
            print(f"   - Features: {len(self.predictor.feature_names)}")
            print(f"   - Classes: {list(self.predictor.class_labels.values())}")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def make_predictions(self):
        """Make predictions on all samples."""
        print(f"\n🔮 Making Predictions on {len(self.features)} samples...")
        
        try:
            start_time = time.time()
            
            # Make predictions in batches for memory efficiency
            batch_size = 1000
            all_predictions = []
            all_labels = []
            all_probabilities = []
            
            for i in range(0, len(self.features), batch_size):
                batch = self.features.iloc[i:i+batch_size]
                result = self.predictor.predict(batch, return_confidence=True)
                
                all_predictions.extend(result['predictions'])
                all_labels.extend(result['prediction_labels'])
                all_probabilities.extend(result['probabilities'])
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"   Processed {i + len(batch)}/{len(self.features)} samples...")
            
            self.predictions = np.array(all_predictions)
            self.prediction_labels = all_labels
            self.probabilities = np.array(all_probabilities)
            
            duration = time.time() - start_time
            print(f"✅ Predictions completed in {duration:.2f} seconds")
            print(f"   - Speed: {len(self.features)/duration:.1f} samples/second")
            
            return True
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return False
    
    def calculate_accuracy(self):
        """Calculate accuracy metrics."""
        print("\n📈 Calculating Accuracy Metrics...")
        
        # Overall accuracy
        accuracy = accuracy_score(self.ground_truth, self.predictions)
        print(f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Per-class accuracy
        print("\n📊 Per-Class Analysis:")
        unique_labels = np.unique(self.ground_truth)
        for label in unique_labels:
            mask = self.ground_truth == label
            class_accuracy = accuracy_score(self.ground_truth[mask], self.predictions[mask])
            class_name = CLASS_LABELS.get(label, f"Class {label}")
            print(f"   {class_name}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
        
        return accuracy
    
    def detailed_analysis(self):
        """Perform detailed analysis of predictions."""
        print("\n🔍 Detailed Analysis...")
        
        # Classification report
        print("\n📋 Classification Report:")
        target_names = [CLASS_LABELS[label] for label in sorted(CLASS_LABELS.keys())]
        report = classification_report(
            self.ground_truth, 
            self.predictions, 
            target_names=target_names,
            zero_division=0
        )
        print(report)
        
        # Confusion matrix
        print("\n📊 Confusion Matrix:")
        cm = confusion_matrix(self.ground_truth, self.predictions)
        print("True\\Pred", end="")
        for label in sorted(CLASS_LABELS.keys()):
            print(f"{CLASS_LABELS[label]:>12}", end="")
        print()
        
        for i, true_label in enumerate(sorted(CLASS_LABELS.keys())):
            print(f"{CLASS_LABELS[true_label]:>12}", end="")
            for j, pred_label in enumerate(sorted(CLASS_LABELS.keys())):
                print(f"{cm[i,j]:>12}", end="")
            print()
        
        # Confidence analysis
        print("\n🎯 Confidence Analysis:")
        confidences = np.max(self.probabilities, axis=1)
        print(f"   Average confidence: {np.mean(confidences):.4f}")
        print(f"   Median confidence: {np.median(confidences):.4f}")
        print(f"   Min confidence: {np.min(confidences):.4f}")
        print(f"   Max confidence: {np.max(confidences):.4f}")
        
        # High confidence predictions
        high_conf_mask = confidences > 0.9
        high_conf_accuracy = accuracy_score(
            self.ground_truth[high_conf_mask], 
            self.predictions[high_conf_mask]
        ) if np.any(high_conf_mask) else 0
        print(f"   High confidence (>0.9) accuracy: {high_conf_accuracy:.4f} ({np.sum(high_conf_mask)} samples)")
        
        # Low confidence predictions
        low_conf_mask = confidences < 0.5
        low_conf_accuracy = accuracy_score(
            self.ground_truth[low_conf_mask], 
            self.predictions[low_conf_mask]
        ) if np.any(low_conf_mask) else 0
        print(f"   Low confidence (<0.5) accuracy: {low_conf_accuracy:.4f} ({np.sum(low_conf_mask)} samples)")
    
    def error_analysis(self):
        """Analyze prediction errors."""
        print("\n❌ Error Analysis...")
        
        # Find misclassified samples
        errors = self.ground_truth != self.predictions
        error_count = np.sum(errors)
        error_rate = error_count / len(self.ground_truth)
        
        print(f"   Total errors: {error_count} ({error_rate*100:.2f}%)")
        
        if error_count > 0:
            # Error distribution by true class
            print("\n   Error distribution by true class:")
            for label in np.unique(self.ground_truth):
                mask = self.ground_truth == label
                class_errors = np.sum(errors[mask])
                class_total = np.sum(mask)
                class_error_rate = class_errors / class_total
                class_name = CLASS_LABELS.get(label, f"Class {label}")
                print(f"     {class_name}: {class_errors}/{class_total} ({class_error_rate*100:.2f}%)")
            
            # Most common error types
            print("\n   Most common error types:")
            error_pairs = list(zip(self.ground_truth[errors], self.predictions[errors]))
            error_counts = Counter(error_pairs)
            for (true_label, pred_label), count in error_counts.most_common(5):
                true_name = CLASS_LABELS.get(true_label, f"Class {true_label}")
                pred_name = CLASS_LABELS.get(pred_label, f"Class {pred_label}")
                print(f"     {true_name} → {pred_name}: {count} errors")
    
    def sample_analysis(self):
        """Analyze specific sample predictions."""
        print("\n🔬 Sample Analysis...")
        
        # Show some correct predictions
        correct_mask = self.ground_truth == self.predictions
        correct_indices = np.where(correct_mask)[0][:5]  # First 5 correct
        
        print("   Correct predictions (first 5):")
        for i, idx in enumerate(correct_indices):
            true_label = CLASS_LABELS.get(self.ground_truth[idx], f"Class {self.ground_truth[idx]}")
            pred_label = CLASS_LABELS.get(self.predictions[idx], f"Class {self.predictions[idx]}")
            confidence = np.max(self.probabilities[idx])
            print(f"     Sample {idx}: {true_label} → {pred_label} (conf: {confidence:.3f})")
        
        # Show some incorrect predictions
        error_mask = self.ground_truth != self.predictions
        error_indices = np.where(error_mask)[0][:5]  # First 5 errors
        
        if len(error_indices) > 0:
            print("\n   Incorrect predictions (first 5):")
            for i, idx in enumerate(error_indices):
                true_label = CLASS_LABELS.get(self.ground_truth[idx], f"Class {self.ground_truth[idx]}")
                pred_label = CLASS_LABELS.get(self.predictions[idx], f"Class {self.predictions[idx]}")
                confidence = np.max(self.probabilities[idx])
                print(f"     Sample {idx}: {true_label} → {pred_label} (conf: {confidence:.3f})")
    
    def run_verification(self):
        """Run the complete accuracy verification."""
        print("🧪 Exo-SDK Accuracy Verification")
        print("=" * 60)
        
        # Load data
        if not self.load_data():
            return False
        
        # Load model
        if not self.load_model():
            return False
        
        # Make predictions
        if not self.make_predictions():
            return False
        
        # Calculate accuracy
        accuracy = self.calculate_accuracy()
        
        # Detailed analysis
        self.detailed_analysis()
        
        # Error analysis
        self.error_analysis()
        
        # Sample analysis
        self.sample_analysis()
        
        # Summary
        total_time = time.time() - self.start_time
        print("\n" + "=" * 60)
        print("📊 VERIFICATION SUMMARY")
        print("=" * 60)
        print(f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"📊 Total Samples: {len(self.training_data)}")
        print(f"⏱️  Processing Time: {total_time:.2f} seconds")
        print(f"⚡ Processing Speed: {len(self.training_data)/total_time:.1f} samples/second")
        
        if accuracy >= 0.8:
            print("🎉 EXCELLENT: Model accuracy is very high!")
        elif accuracy >= 0.7:
            print("✅ GOOD: Model accuracy is good!")
        elif accuracy >= 0.6:
            print("⚠️  FAIR: Model accuracy is acceptable but could be improved.")
        else:
            print("❌ POOR: Model accuracy needs improvement.")
        
        return True

def main():
    """Run the accuracy verification."""
    verifier = AccuracyVerification()
    success = verifier.run_verification()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
