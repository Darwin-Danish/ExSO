# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

The **Exo-SDK** is an exoplanet candidate classification toolkit that uses a V2 stacking ensemble model (LightGBM + XGBoost + CatBoost) to classify exoplanet candidates into three classes: False Positive, Candidate, and Positive. The SDK provides data preprocessing, feature engineering, model training, and prediction capabilities.

## Development Commands

### Setup and Installation
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate     # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Testing
```bash
# Run complete test suite
cd tests/
python run_tests.py

# Run individual tests
python test_basic.py                    # Basic functionality test
python test_complete_sdk.py            # Comprehensive SDK test
python test_accuracy_verification.py   # Model accuracy verification
python demo_pypi_package.py           # PyPI package test
python model_import.py                 # Model import test

# Quick start example
python quick_start.py
```

### SDK Usage
```bash
# Quick prediction example
python quick_start.py

# Test SDK functionality
python -c "
from exso_sdk.model import ExoplanetPredictor
from exso_sdk.preprocessing import create_sample_data

predictor = ExoplanetPredictor()
sample = create_sample_data()
result = predictor.predict(sample, return_confidence=True)
print(f'Prediction: {result[\"prediction_labels\"][0]}')
print(f'Confidence: {result[\"confidence_scores\"][0]:.2f}')
"
```

### Package Management
```bash
# Build package
python -m build

# Upload to PyPI
python upload_to_pypi.py

# Test PyPI installation
pip install --index-url https://test.pypi.org/simple/ exso-sdk
```

### Single Prediction Example
```bash
# Test single prediction
python -c "
from exso_sdk.model import ExoplanetPredictor
from exso_sdk.preprocessing import create_sample_data

predictor = ExoplanetPredictor()
sample = create_sample_data()
result = predictor.predict(sample, return_confidence=True)
print(f'Prediction: {result[\"prediction_labels\"][0]}')
print(f'Confidence: {result[\"confidence_scores\"][0]:.2f}')
"
```

## Architecture Overview

### Core Architecture Pattern
The SDK follows a modular architecture with clear separation of concerns:

**Model Pipeline Flow:**
1. **Data Layer** (`data.py`) - Handles CSV loading, validation, dataset merging
2. **Preprocessing Layer** (`preprocessing.py`) - Data cleaning, scaling, NaN handling  
3. **Feature Engineering** (`features.py`) - Domain-specific and statistical features
4. **Model Layer** (`model.py`) - V2 stacking ensemble (LightGBM/XGBoost/CatBoost)

### V2 Model Architecture
The V2 model uses a sophisticated stacking ensemble approach:
- **Base Models:** LightGBM, XGBoost, CatBoost classifiers
- **Meta-learner:** Final stacking classifier combining base model predictions
- **Preprocessing:** Column transformer with different strategies for critical vs auxiliary features
- **NaN Handling:** Robust imputation strategies for missing values
- **Feature Importance:** Available through base model inspection

### Key Components

#### Configuration (`config.py`)
- **REQUIRED_COLUMNS:** 16 features split into CRITICAL_FEATURES (7) and AUXILIARY_FEATURES (9)
- **CLASS_LABELS:** Maps encoded labels (-1, 0, 1) to human-readable classes
- **MODEL_CONFIG:** Defines model type, version, and preprocessing pipeline
- **Path Management:** Handles model/encoder paths for both package and development modes

#### Model Interface (`model.py`)
- **ExoplanetPredictor:** Main prediction class with V2 model interface
- **Batch Prediction:** Handles DataFrames, numpy arrays, and lists
- **Single Prediction:** Convenience method for individual samples
- **Feature Importance:** Extracts importance from first base model (LightGBM)
- **Confidence Scoring:** Returns max probability as confidence measure

#### Data Processing Pipeline
The preprocessing follows a specific order:
1. **Validation:** Check required columns and data types (`data.validate_dataset()`)
2. **Cleaning:** Handle missing values with strategy-based approach
3. **Feature Engineering:** Compute period, statistical, and domain features
4. **Scaling/Encoding:** Apply V2 preprocessing pipeline with NaN handling
5. **Prediction:** Feed through stacking ensemble model

### Important Design Patterns

#### Environment-Based Configuration
The SDK uses environment variables for flexible deployment:
- `EXSO_MODEL_PATH`: Override default model path
- `EXSO_LABEL_ENCODER_PATH`: Override default encoder path

#### Dual Path Support
The configuration system supports both:
- **Package Mode:** Models in `exso_sdk/model_data/v2/`
- **Development Mode:** Models in `model/v2/` (relative to package root)

#### Error Handling Strategy
- **Validation Errors:** Early detection of missing columns, invalid data types
- **Model Errors:** Runtime exception handling with informative messages

### Testing Strategy

The test suite follows a comprehensive approach:
- **Unit Tests:** Individual component testing (`test_basic.py`)
- **Integration Tests:** Full pipeline testing (`test_complete_sdk.py`)
- **Accuracy Tests:** Model performance verification (`test_accuracy_verification.py`)
- **Package Tests:** PyPI installation and import testing (`demo_pypi_package.py`)

### Development Workflow Considerations

#### Feature Requirements
All input data must contain the 16 required columns defined in `REQUIRED_COLUMNS`. Missing any of these will cause validation errors.

#### Model Versioning
The system is designed around V2 models. When adding new model versions:
1. Update `MODEL_CONFIG['version']`
2. Add new paths in `config.py`
3. Update `ExoplanetPredictor` to handle version-specific logic
4. Maintain backward compatibility in SDK responses

#### SDK Design Philosophy
The SDK follows clean architecture principles with:
- **Modular design:** Clear separation of concerns
- **Easy-to-use interface:** Simple prediction methods
- **Batch processing:** Efficient handling of multiple samples
- **Comprehensive error handling:** Informative error messages

#### Dependency Management
The project uses explicit version constraints for ML dependencies (LightGBM, XGBoost, CatBoost) to ensure model compatibility across environments.

## Environment Variables

- `EXSO_MODEL_PATH`: Path to custom V2 model file
- `EXSO_LABEL_ENCODER_PATH`: Path to custom label encoder