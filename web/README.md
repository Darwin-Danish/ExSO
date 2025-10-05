# Exo-Operator Web Interface

A professional web interface for the Exo-Operator exoplanet classification SDK, designed for both researchers and novices in the field.

## Features

### 🎯 Dual Mode Interface
- **Simple Mode**: Beginner-friendly interface with essential parameters
- **Pro Mode**: Advanced interface with full parameter control and hyperparameter tuning

### 🔬 Core Functionality
- **Real-time Prediction**: Classify exoplanet candidates instantly
- **Batch Analysis**: Upload and analyze entire datasets
- **Model Training**: Train and fine-tune models with custom data
- **Performance Metrics**: View model accuracy and feature importance
- **Data Visualization**: Interactive charts and result displays

### 🚀 Advanced Features
- **Hyperparameter Tuning**: Adjust model parameters for optimal performance
- **Data Upload**: Support for CSV, Excel, and JSON formats
- **Model Statistics**: Comprehensive accuracy metrics and performance analysis
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Exo-SDK is Available**:
   The web interface requires the Exo-SDK to be installed. Make sure it's available in your Python environment.

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **Access the Interface**:
   Open your browser and navigate to `http://localhost:2429`

## Usage

### Simple Mode
Perfect for beginners who want to quickly classify exoplanet candidates:

1. Navigate to the **Predict** section
2. Enter basic parameters:
   - Orbital Period (days)
   - Transit Depth (ppm)
   - Planet Radius (Earth radii)
   - Semi-major Axis (AU)
3. Click **Classify Exoplanet**
4. View the prediction results with confidence scores

### Pro Mode
Designed for researchers and advanced users:

1. Toggle to **Pro Mode** using the mode switch
2. Access all 16 required parameters for precise classification
3. Upload datasets for batch analysis
4. Fine-tune hyperparameters for model training
5. View detailed performance metrics and feature importance

### Data Analysis
1. Go to the **Analyze** section
2. Upload a CSV, Excel, or JSON file with exoplanet data
3. Preview the data structure and validate format
4. Run batch analysis to classify all candidates
5. View detailed results with probability breakdowns

### Model Training
1. Navigate to the **Train** section
2. Upload labeled training data
3. Choose between Simple or Pro training modes
4. Adjust hyperparameters (Pro mode only)
5. Monitor training progress
6. Deploy the trained model

## API Endpoints

The web interface provides a RESTful API:

- `GET /api/health` - Health check and model status
- `GET /api/model/info` - Model information and configuration
- `GET /api/model/accuracy` - Model performance metrics
- `GET /api/model/features` - Feature importance rankings
- `POST /api/predict` - Single prediction
- `POST /api/upload` - File upload and validation
- `POST /api/predict/batch` - Batch prediction from uploaded file
- `GET /api/hyperparameters` - Get current hyperparameters
- `POST /api/hyperparameters` - Update hyperparameters
- `POST /api/train` - Start model training

## File Formats

### Supported Input Formats
- **CSV**: Comma-separated values with headers
- **Excel**: .xlsx files with data sheets
- **JSON**: JSON arrays or objects with exoplanet data

### Required Columns
The following columns are required for prediction:
- `koi_period` - Orbital period in days
- `koi_depth` - Transit depth in ppm
- `koi_prad` - Planet radius in Earth radii
- `koi_sma` - Semi-major axis in AU
- `koi_teq` - Equilibrium temperature in K
- `koi_insol` - Insolation flux in Earth units
- `koi_model_snr` - Model signal-to-noise ratio
- `koi_time0bk` - Transit epoch in BKJD
- `koi_duration` - Transit duration in hours
- `koi_incl` - Inclination in degrees
- `koi_srho` - Stellar density in g/cm³
- `koi_srad` - Stellar radius in Solar radii
- `koi_smass` - Stellar mass in Solar masses
- `koi_steff` - Stellar effective temperature in K
- `koi_slogg` - Stellar surface gravity in log g
- `koi_smet` - Stellar metallicity in dex

## Architecture

### Frontend
- **HTML5**: Semantic markup with modern structure
- **CSS3**: Professional styling with CSS Grid and Flexbox
- **JavaScript**: Vanilla JS with ES6+ features
- **Responsive Design**: Mobile-first approach

### Backend
- **Flask**: Lightweight Python web framework
- **RESTful API**: Clean API design with JSON responses
- **File Upload**: Secure file handling with validation
- **Error Handling**: Comprehensive error management

### Integration
- **Exo-SDK**: Direct integration with the exoplanet classification SDK
- **Model Loading**: Automatic model initialization and health checks
- **Data Validation**: Robust input validation and error reporting

## Development

### Project Structure
```
web/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── index.html        # Main HTML template
└── static/
    ├── css/
    │   └── style.css     # Main stylesheet
    └── js/
        └── app.js        # JavaScript application
```

### Customization
- **Styling**: Modify `static/css/style.css` for custom themes
- **Functionality**: Extend `static/js/app.js` for new features
- **API**: Add new endpoints in `app.py`
- **Templates**: Update `templates/index.html` for UI changes

## Troubleshooting

### Common Issues

1. **Model Not Loading**:
   - Ensure Exo-SDK is properly installed
   - Check that model files are in the correct location
   - Verify Python path includes the SDK

2. **File Upload Errors**:
   - Check file format (CSV, Excel, JSON only)
   - Ensure required columns are present
   - Verify file size is under 50MB limit

3. **Prediction Failures**:
   - Validate input data format
   - Check for missing required parameters
   - Ensure numeric values are properly formatted

### Debug Mode
Run the application in debug mode for detailed error information:
```bash
export FLASK_DEBUG=1
python app.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the main project LICENSE file for details.

## Support

For support and questions:
- Check the documentation
- Review the API endpoints
- Open an issue on GitHub
- Contact the development team

---

**Exo-Operator Web Interface** - Making exoplanet classification accessible to everyone.
