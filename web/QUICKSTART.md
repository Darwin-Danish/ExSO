# Exo-Operator Web Interface - Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies
```bash
# Navigate to the web directory
cd web

# Install web interface requirements
pip install -r requirements.txt

# Make sure Exo-SDK is installed
cd ..
pip install -e .
cd web
```

### 2. Start the Application
```bash
# Option 1: Use the startup script (recommended)
python run.py

# Option 2: Run directly
python app.py
```

### 3. Open Your Browser
Navigate to: **http://localhost:2429**

## 🎯 Quick Demo

### Simple Mode (Beginners)
1. Go to **Predict** section
2. Enter these sample values:
   - Orbital Period: `3.5` days
   - Transit Depth: `1200` ppm
   - Planet Radius: `1.2` Earth radii
   - Semi-major Axis: `0.045` AU
3. Click **Classify Exoplanet**
4. View the prediction result!

### Pro Mode (Researchers)
1. Toggle to **Pro Mode** (top right)
2. Go to **Analyze** section
3. Upload the sample file: `sample_data/sample_exoplanets.csv`
4. Click **Analyze Dataset**
5. View detailed batch analysis results!

### Model Training
1. Go to **Train** section
2. Upload training data: `sample_data/training_data.csv`
3. Choose Simple or Pro training mode
4. Start training and monitor progress!

## 🔧 Troubleshooting

### "Model not available" error
```bash
# Make sure you're in the project root
cd /path/to/Exo-Operator

# Install the SDK
pip install -e .

# Then start the web interface
cd web
python run.py
```

### Port already in use
```bash
# Kill any process using port 2429
lsof -ti:2429 | xargs kill -9

# Or use a different port
export FLASK_RUN_PORT=5001
python app.py
```

### Missing dependencies
```bash
# Install all requirements
pip install -r requirements.txt

# If still having issues, try upgrading pip
pip install --upgrade pip
pip install -r requirements.txt
```

## 📊 Sample Data

The `sample_data/` directory contains:
- `sample_exoplanets.csv` - Sample data for prediction testing
- `training_data.csv` - Labeled data for model training

## 🎨 Features Overview

### Dashboard
- Model status and health check
- Accuracy metrics (94% overall accuracy)
- Feature importance rankings
- Quick action buttons

### Prediction
- **Simple Mode**: 4 essential parameters
- **Pro Mode**: All 16 parameters
- Real-time classification
- Confidence scores and probability breakdowns

### Analysis
- File upload (CSV, Excel, JSON)
- Data validation and preview
- Batch prediction for entire datasets
- Detailed results with visualizations

### Training
- **Simple Mode**: Automatic training
- **Pro Mode**: Hyperparameter tuning
- Progress monitoring
- Model performance tracking

## 🔗 API Endpoints

- `GET /api/health` - System status
- `GET /api/model/info` - Model details
- `POST /api/predict` - Single prediction
- `POST /api/upload` - File upload
- `POST /api/predict/batch` - Batch analysis
- `POST /api/train` - Model training

## 📱 Mobile Support

The interface is fully responsive and works on:
- Desktop computers
- Tablets
- Mobile phones

## 🎓 Learning Resources

- **For Beginners**: Start with Simple Mode predictions
- **For Researchers**: Use Pro Mode for advanced analysis
- **For Developers**: Check the API documentation
- **For Data Scientists**: Explore hyperparameter tuning

## 🆘 Need Help?

1. Check the console output for error messages
2. Verify all dependencies are installed
3. Ensure the Exo-SDK is properly installed
4. Check that port 5000 is available
5. Review the full README.md for detailed documentation

---

**Ready to explore exoplanets? Start the application and begin classifying!** 🌟
