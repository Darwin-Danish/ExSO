#!/usr/bin/env python3
"""
Test script to verify dependencies and run the app
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

def test_imports():
    """Test if all required modules can be imported."""
    try:
        import flask
        print("✓ Flask imported successfully")
    except ImportError as e:
        print(f"✗ Flask import failed: {e}")
        return False
    
    try:
        import flask_cors
        print("✓ Flask-CORS imported successfully")
    except ImportError as e:
        print(f"✗ Flask-CORS import failed: {e}")
        return False
    
    try:
        import pandas
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import numpy
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import exso_sdk
        print("✓ Exo-SDK imported successfully")
    except ImportError as e:
        print(f"✗ Exo-SDK import failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing Exo-Operator Web Interface Dependencies")
    print("=" * 50)
    
    if test_imports():
        print("\n✓ All dependencies are available!")
        print("Starting the application...")
        
        try:
            from app import app, init_predictor
            init_predictor()
            print("🚀 Starting on http://localhost:2429")
            app.run(debug=True, host='0.0.0.0', port=2429)
        except Exception as e:
            print(f"✗ Error starting app: {e}")
            sys.exit(1)
    else:
        print("\n✗ Some dependencies are missing!")
        print("Please install them with:")
        print("  uv add flask flask-cors pandas numpy openpyxl werkzeug")
        sys.exit(1)
