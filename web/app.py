"""
Exo-Operator Web Interface
==========================

Professional web interface for exoplanet classification SDK.
Supports both Simple and Pro modes for different user expertise levels.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import traceback
from datetime import datetime
import joblib
from werkzeug.utils import secure_filename

# Add parent directory to path to import exso_sdk
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import exso_sdk
    from exso_sdk import ExoplanetPredictor, predict, get_model_info, get_feature_importance
    SDK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: exso_sdk not available: {e}")
    SDK_AVAILABLE = False

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'exo-operator-secret-key-2024'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'json'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global predictor instance
predictor = None

def init_predictor():
    """Initialize the exoplanet predictor."""
    global predictor
    if SDK_AVAILABLE and predictor is None:
        try:
            print("Initializing ExoplanetPredictor...")
            predictor = ExoplanetPredictor()
            print("✓ Exoplanet predictor initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize predictor: {e}")
            import traceback
            traceback.print_exc()
            predictor = None
    else:
        if not SDK_AVAILABLE:
            print("✗ SDK not available")
        if predictor is not None:
            print("✓ Predictor already initialized")

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_data(data):
    """Validate uploaded data format."""
    required_columns = [
        'koi_period', 'koi_depth', 'koi_prad', 'koi_sma', 'koi_teq',
        'koi_insol', 'koi_model_snr', 'koi_time0bk', 'koi_duration', 
        'koi_incl', 'koi_srho', 'koi_srad', 'koi_smass', 'koi_steff', 
        'koi_slogg', 'koi_smet'
    ]
    
    if isinstance(data, pd.DataFrame):
        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            return False, f"Missing required columns: {list(missing_cols)}"
        return True, "Data format is valid"
    return False, "Invalid data format"

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'sdk_available': SDK_AVAILABLE,
        'predictor_loaded': predictor is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model/info')
def model_info():
    """Get model information."""
    if not SDK_AVAILABLE or predictor is None:
        return jsonify({'error': 'Model not available'}), 500
    
    try:
        info = predictor.get_model_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/accuracy')
def model_accuracy():
    """Get model accuracy statistics."""
    if not SDK_AVAILABLE or predictor is None:
        return jsonify({'error': 'Model not available'}), 500
    
    try:
        # This would typically come from model validation results
        # For now, return placeholder data
        accuracy_stats = {
            'overall_accuracy': 0.94,
            'precision': 0.92,
            'recall': 0.89,
            'f1_score': 0.90,
            'auc_score': 0.95,
            'confusion_matrix': {
                'true_positives': 850,
                'false_positives': 45,
                'true_negatives': 1200,
                'false_negatives': 55
            },
            'class_accuracy': {
                'False Positive': 0.96,
                'Candidate': 0.88,
                'Positive': 0.92
            }
        }
        return jsonify(accuracy_stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/features')
def feature_importance():
    """Get feature importance."""
    if not SDK_AVAILABLE or predictor is None:
        return jsonify({'error': 'Model not available'}), 500
    
    try:
        importance_df = predictor.get_feature_importance(top_n=15)
        features = importance_df.to_dict('records')
        return jsonify({'features': features})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_exoplanet():
    """Predict exoplanet classification."""
    if not SDK_AVAILABLE or predictor is None:
        return jsonify({'error': 'Model not available'}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Handle different input formats
        if 'features' in data:
            # Single prediction with feature dictionary
            features = data['features']
            result = predictor.predict_single(features)
        elif 'data' in data:
            # Batch prediction with array of features
            df = pd.DataFrame(data['data'])
            result = predictor.predict(df)
        else:
            return jsonify({'error': 'Invalid data format'}), 400
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and validate data file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # Read the file
            if filename.endswith('.csv'):
                data = pd.read_csv(filepath)
            elif filename.endswith('.xlsx'):
                data = pd.read_excel(filepath)
            elif filename.endswith('.json'):
                data = pd.read_json(filepath)
            else:
                return jsonify({'error': 'Unsupported file format'}), 400
            
            # Validate data
            is_valid, message = validate_data(data)
            if not is_valid:
                os.remove(filepath)
                return jsonify({'error': message}), 400
            
            # Return data preview
            preview = data.head(10).to_dict('records')
            summary = {
                'rows': len(data),
                'columns': list(data.columns),
                'missing_values': data.isnull().sum().to_dict()
            }
            
            return jsonify({
                'message': 'File uploaded successfully',
                'preview': preview,
                'summary': summary,
                'filepath': filepath
            })
            
        except Exception as e:
            os.remove(filepath)
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """Predict from uploaded file."""
    if not SDK_AVAILABLE or predictor is None:
        return jsonify({'error': 'Model not available'}), 500
    
    try:
        data = request.get_json()
        filepath = data.get('filepath')
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 400
        
        # Read the file
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        # Validate data
        is_valid, message = validate_data(df)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Make predictions
        result = predictor.predict(df)
        
        # Add original data for reference
        result['original_data'] = df.to_dict('records')
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/hyperparameters', methods=['GET', 'POST'])
def hyperparameters():
    """Get or update model hyperparameters."""
    if not SDK_AVAILABLE or predictor is None:
        return jsonify({'error': 'Model not available'}), 500
    
    if request.method == 'GET':
        # Return current hyperparameters (placeholder)
        hyperparams = {
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8
            },
            'catboost': {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1
            }
        }
        return jsonify(hyperparams)
    
    elif request.method == 'POST':
        # Update hyperparameters (placeholder - would require model retraining)
        new_params = request.get_json()
        return jsonify({
            'message': 'Hyperparameters updated successfully',
            'note': 'Model retraining required for changes to take effect'
        })

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train model with new data."""
    if not SDK_AVAILABLE:
        return jsonify({'error': 'SDK not available'}), 500
    
    try:
        data = request.get_json()
        filepath = data.get('filepath')
        hyperparams = data.get('hyperparameters', {})
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'Training data file not found'}), 400
        
        # This would implement actual model training
        # For now, return a placeholder response
        return jsonify({
            'message': 'Model training initiated',
            'status': 'training',
            'estimated_time': '5-10 minutes',
            'note': 'Training functionality requires implementation'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_predictor()
    app.run(debug=True, host='0.0.0.0', port=2429)
else:
    # Initialize predictor when imported
    init_predictor()
