/**
 * Exo-Operator Web Interface JavaScript
 * Professional exoplanet classification platform
 */

class ExoOperatorApp {
    constructor() {
        this.currentMode = 'simple';
        this.currentSection = 'dashboard';
        this.uploadedFile = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadDashboard();
        this.checkHealth();
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const section = link.dataset.section;
                this.switchSection(section);
            });
        });

        // Mode toggle
        document.getElementById('modeToggle').addEventListener('click', () => {
            this.toggleMode();
        });

        // Prediction forms
        document.getElementById('simplePredictForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleSimplePrediction();
        });

        document.getElementById('proPredictForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleProPrediction();
        });

        // File upload
        document.getElementById('fileInput').addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files[0]);
        });

        document.getElementById('trainFileInput').addEventListener('change', (e) => {
            this.handleTrainingFileUpload(e.target.files[0]);
        });

        document.getElementById('proTrainFileInput').addEventListener('change', (e) => {
            this.handleProTrainingFileUpload(e.target.files[0]);
        });

        // Drag and drop
        this.setupDragAndDrop();
    }

    setupDragAndDrop() {
        const uploadAreas = document.querySelectorAll('.upload-area');
        
        uploadAreas.forEach(area => {
            area.addEventListener('dragover', (e) => {
                e.preventDefault();
                area.classList.add('dragover');
            });

            area.addEventListener('dragleave', (e) => {
                e.preventDefault();
                area.classList.remove('dragover');
            });

            area.addEventListener('drop', (e) => {
                e.preventDefault();
                area.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    this.handleFileUpload(files[0]);
                }
            });
        });
    }

    async checkHealth() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();
            
            const statusElement = document.getElementById('modelStatus');
            if (data.predictor_loaded) {
                statusElement.innerHTML = `
                    <div class="status-dot ready"></div>
                    <span>Model Ready</span>
                `;
            } else {
                statusElement.innerHTML = `
                    <div class="status-dot error"></div>
                    <span>Model Error</span>
                `;
            }
        } catch (error) {
            console.error('Health check failed:', error);
            const statusElement = document.getElementById('modelStatus');
            statusElement.innerHTML = `
                <div class="status-dot error"></div>
                <span>Connection Error</span>
            `;
        }
    }

    async loadDashboard() {
        try {
            // Load model info
            const modelResponse = await fetch('/api/model/info');
            if (modelResponse.ok) {
                const modelInfo = await modelResponse.json();
                this.updateModelInfo(modelInfo);
            }

            // Load accuracy metrics
            const accuracyResponse = await fetch('/api/model/accuracy');
            if (accuracyResponse.ok) {
                const accuracy = await accuracyResponse.json();
                this.updateAccuracyMetrics(accuracy);
            }

            // Load feature importance
            const featuresResponse = await fetch('/api/model/features');
            if (featuresResponse.ok) {
                const features = await featuresResponse.json();
                this.updateFeatureImportance(features.features);
            }
        } catch (error) {
            console.error('Failed to load dashboard:', error);
            this.showToast('Failed to load dashboard data', 'error');
        }
    }

    updateModelInfo(info) {
        document.getElementById('modelVersion').textContent = info.version || 'N/A';
        document.getElementById('modelType').textContent = info.type || 'N/A';
        document.getElementById('modelFeatures').textContent = info.features || 'N/A';
        document.getElementById('modelInfo').style.display = 'block';
    }

    updateAccuracyMetrics(metrics) {
        document.getElementById('overallAccuracy').textContent = `${(metrics.overall_accuracy * 100).toFixed(1)}%`;
        document.getElementById('precision').textContent = `${(metrics.precision * 100).toFixed(1)}%`;
        document.getElementById('recall').textContent = `${(metrics.recall * 100).toFixed(1)}%`;
        document.getElementById('f1Score').textContent = `${(metrics.f1_score * 100).toFixed(1)}%`;
    }

    updateFeatureImportance(features) {
        const container = document.getElementById('featureList');
        container.innerHTML = '';

        features.forEach(feature => {
            const item = document.createElement('div');
            item.className = 'feature-item';
            
            const importancePercent = (feature.importance * 100).toFixed(1);
            
            item.innerHTML = `
                <span class="feature-name">${feature.feature}</span>
                <div class="feature-importance">
                    <div class="importance-bar">
                        <div class="importance-fill" style="width: ${importancePercent}%"></div>
                    </div>
                    <span class="importance-value">${importancePercent}%</span>
                </div>
            `;
            
            container.appendChild(item);
        });
    }

    switchSection(sectionName) {
        // Update navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        document.querySelector(`[data-section="${sectionName}"]`).classList.add('active');

        // Update sections
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active');
        });
        document.getElementById(sectionName).classList.add('active');

        this.currentSection = sectionName;

        // Load section-specific data
        if (sectionName === 'dashboard') {
            this.loadDashboard();
        }
    }

    toggleMode() {
        this.currentMode = this.currentMode === 'simple' ? 'pro' : 'simple';
        
        const toggleBtn = document.getElementById('modeToggle');
        const modeText = toggleBtn.querySelector('.mode-text');
        const modeIcon = toggleBtn.querySelector('i');
        
        if (this.currentMode === 'pro') {
            toggleBtn.classList.add('pro-mode');
            modeText.textContent = 'Pro';
            modeIcon.className = 'fas fa-toggle-on';
        } else {
            toggleBtn.classList.remove('pro-mode');
            modeText.textContent = 'Simple';
            modeIcon.className = 'fas fa-toggle-off';
        }

        // Update UI based on mode
        this.updateModeUI();
    }

    updateModeUI() {
        // Update prediction forms
        const simplePredict = document.getElementById('simplePredict');
        const proPredict = document.getElementById('proPredict');
        const simpleTrain = document.getElementById('simpleTrain');
        const proTrain = document.getElementById('proTrain');

        if (this.currentMode === 'simple') {
            simplePredict.style.display = 'block';
            proPredict.style.display = 'none';
            simpleTrain.style.display = 'block';
            proTrain.style.display = 'none';
        } else {
            simplePredict.style.display = 'none';
            proPredict.style.display = 'block';
            simpleTrain.style.display = 'none';
            proTrain.style.display = 'block';
        }
    }

    async handleSimplePrediction() {
        const form = document.getElementById('simplePredictForm');
        const formData = new FormData(form);
        
        const features = {
            koi_period: parseFloat(formData.get('koi_period')),
            koi_depth: parseFloat(formData.get('koi_depth')),
            koi_prad: parseFloat(formData.get('koi_prad')),
            koi_sma: parseFloat(formData.get('koi_sma')),
            // Fill remaining required fields with default values
            koi_teq: 300,
            koi_insol: 1.0,
            koi_model_snr: 10.0,
            koi_time0bk: 1000,
            koi_duration: 2.0,
            koi_incl: 90.0,
            koi_srho: 1.0,
            koi_srad: 1.0,
            koi_smass: 1.0,
            koi_steff: 5778,
            koi_slogg: 4.4,
            koi_smet: 0.0
        };

        await this.makePrediction(features);
    }

    async handleProPrediction() {
        const form = document.getElementById('proPredictForm');
        const formData = new FormData(form);
        
        const features = {};
        for (const [key, value] of formData.entries()) {
            features[key] = parseFloat(value);
        }

        await this.makePrediction(features);
    }

    async makePrediction(features) {
        this.showLoading('Making prediction...');

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ features })
            });

            const result = await response.json();

            if (response.ok) {
                this.displayPredictionResults(result);
                this.showToast('Prediction completed successfully!', 'success');
            } else {
                throw new Error(result.error || 'Prediction failed');
            }
        } catch (error) {
            console.error('Prediction error:', error);
            this.showToast(`Prediction failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    displayPredictionResults(result) {
        const resultsContainer = document.getElementById('predictionResults');
        const resultContent = document.getElementById('resultContent');

        const prediction = result.prediction_labels[0];
        const confidence = result.confidence_scores[0];
        const probabilities = result.probabilities[0];

        resultContent.innerHTML = `
            <div class="result-card">
                <div class="result-title">Classification Result</div>
                <div class="result-value">${prediction}</div>
                <div class="result-confidence">
                    <span>Confidence: ${(confidence * 100).toFixed(1)}%</span>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidence * 100}%"></div>
                    </div>
                </div>
            </div>
            
            <div class="result-card">
                <div class="result-title">Probability Breakdown</div>
                <div class="probability-list">
                    ${probabilities.map((prob, index) => {
                        const labels = ['False Positive', 'Candidate', 'Positive'];
                        return `
                            <div class="probability-item">
                                <span>${labels[index]}:</span>
                                <span>${(prob * 100).toFixed(1)}%</span>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        `;

        resultsContainer.style.display = 'block';
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }

    async handleFileUpload(file) {
        if (!file) return;

        this.showLoading('Uploading file...');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                this.uploadedFile = result;
                this.displayFilePreview(result);
                this.showToast('File uploaded successfully!', 'success');
            } else {
                throw new Error(result.error || 'Upload failed');
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showToast(`Upload failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    displayFilePreview(data) {
        const preview = document.getElementById('filePreview');
        const summary = document.getElementById('fileSummary');
        const fileData = document.getElementById('fileData');

        summary.innerHTML = `
            <div class="summary-item">
                <strong>Rows:</strong> ${data.summary.rows}
            </div>
            <div class="summary-item">
                <strong>Columns:</strong> ${data.summary.columns.length}
            </div>
            <div class="summary-item">
                <strong>Missing Values:</strong> ${Object.values(data.summary.missing_values).reduce((a, b) => a + b, 0)}
            </div>
        `;

        // Create table for data preview
        const table = document.createElement('table');
        table.className = 'data-table';
        
        // Header
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        data.summary.columns.forEach(col => {
            const th = document.createElement('th');
            th.textContent = col;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Data rows
        const tbody = document.createElement('tbody');
        data.preview.forEach(row => {
            const tr = document.createElement('tr');
            data.summary.columns.forEach(col => {
                const td = document.createElement('td');
                td.textContent = row[col] || '-';
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);

        fileData.innerHTML = '';
        fileData.appendChild(table);

        preview.style.display = 'block';
    }

    clearFile() {
        this.uploadedFile = null;
        document.getElementById('filePreview').style.display = 'none';
        document.getElementById('fileInput').value = '';
    }

    async analyzeFile() {
        if (!this.uploadedFile) {
            this.showToast('Please upload a file first', 'warning');
            return;
        }

        this.showLoading('Analyzing dataset...');

        try {
            const response = await fetch('/api/predict/batch', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ filepath: this.uploadedFile.filepath })
            });

            const result = await response.json();

            if (response.ok) {
                this.displayAnalysisResults(result);
                this.showToast('Analysis completed successfully!', 'success');
            } else {
                throw new Error(result.error || 'Analysis failed');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            this.showToast(`Analysis failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    displayAnalysisResults(result) {
        const resultsContainer = document.getElementById('analysisResults');
        const analysisContent = document.getElementById('analysisContent');

        const totalPredictions = result.predictions.length;
        const predictions = result.prediction_labels;
        
        // Count predictions by class
        const classCounts = predictions.reduce((acc, pred) => {
            acc[pred] = (acc[pred] || 0) + 1;
            return acc;
        }, {});

        analysisContent.innerHTML = `
            <div class="analysis-summary">
                <h4>Analysis Summary</h4>
                <div class="summary-stats">
                    <div class="stat-item">
                        <span class="stat-label">Total Predictions:</span>
                        <span class="stat-value">${totalPredictions}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Average Confidence:</span>
                        <span class="stat-value">${(result.avg_confidence * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>
            
            <div class="class-distribution">
                <h4>Classification Distribution</h4>
                <div class="distribution-chart">
                    ${Object.entries(classCounts).map(([className, count]) => {
                        const percentage = (count / totalPredictions * 100).toFixed(1);
                        return `
                            <div class="distribution-item">
                                <div class="distribution-label">${className}</div>
                                <div class="distribution-bar">
                                    <div class="distribution-fill" style="width: ${percentage}%"></div>
                                </div>
                                <div class="distribution-value">${count} (${percentage}%)</div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
            
            <div class="detailed-results">
                <h4>Detailed Results</h4>
                <div class="results-table-container">
                    <table class="results-table">
                        <thead>
                            <tr>
                                <th>Index</th>
                                <th>Prediction</th>
                                <th>Confidence</th>
                                <th>False Positive</th>
                                <th>Candidate</th>
                                <th>Positive</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${result.predictions.map((pred, index) => `
                                <tr>
                                    <td>${index + 1}</td>
                                    <td>${result.prediction_labels[index]}</td>
                                    <td>${(result.confidence_scores[index] * 100).toFixed(1)}%</td>
                                    <td>${(result.probabilities[index][0] * 100).toFixed(1)}%</td>
                                    <td>${(result.probabilities[index][1] * 100).toFixed(1)}%</td>
                                    <td>${(result.probabilities[index][2] * 100).toFixed(1)}%</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;

        resultsContainer.style.display = 'block';
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }

    async handleTrainingFileUpload(file) {
        if (!file) return;

        this.showLoading('Uploading training file...');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                this.trainingFile = result;
                this.displayTrainingPreview(result);
                this.showToast('Training file uploaded successfully!', 'success');
            } else {
                throw new Error(result.error || 'Upload failed');
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showToast(`Upload failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    displayTrainingPreview(data) {
        const preview = document.getElementById('trainPreview');
        const summary = document.getElementById('trainSummary');

        summary.innerHTML = `
            <div class="summary-item">
                <strong>Rows:</strong> ${data.summary.rows}
            </div>
            <div class="summary-item">
                <strong>Columns:</strong> ${data.summary.columns.length}
            </div>
            <div class="summary-item">
                <strong>Missing Values:</strong> ${Object.values(data.summary.missing_values).reduce((a, b) => a + b, 0)}
            </div>
        `;

        preview.style.display = 'block';
    }

    async handleProTrainingFileUpload(file) {
        // Same as regular training file upload
        await this.handleTrainingFileUpload(file);
    }

    async startSimpleTraining() {
        if (!this.trainingFile) {
            this.showToast('Please upload a training file first', 'warning');
            return;
        }

        await this.startTraining({});
    }

    async startProTraining() {
        if (!this.trainingFile) {
            this.showToast('Please upload a training file first', 'warning');
            return;
        }

        const hyperparams = {
            lightgbm: {
                n_estimators: parseInt(document.getElementById('lgb_n_estimators').value),
                max_depth: parseInt(document.getElementById('lgb_max_depth').value),
                learning_rate: parseFloat(document.getElementById('lgb_learning_rate').value)
            },
            xgboost: {
                n_estimators: parseInt(document.getElementById('xgb_n_estimators').value),
                max_depth: parseInt(document.getElementById('xgb_max_depth').value),
                learning_rate: parseFloat(document.getElementById('xgb_learning_rate').value)
            },
            catboost: {
                iterations: parseInt(document.getElementById('cat_iterations').value),
                depth: parseInt(document.getElementById('cat_depth').value),
                learning_rate: parseFloat(document.getElementById('cat_learning_rate').value)
            }
        };

        await this.startTraining(hyperparams);
    }

    async startTraining(hyperparams) {
        this.showLoading('Starting training...');

        try {
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    filepath: this.trainingFile.filepath,
                    hyperparameters: hyperparams
                })
            });

            const result = await response.json();

            if (response.ok) {
                this.showTrainingProgress(result);
                this.showToast('Training started successfully!', 'success');
            } else {
                throw new Error(result.error || 'Training failed to start');
            }
        } catch (error) {
            console.error('Training error:', error);
            this.showToast(`Training failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    showTrainingProgress(result) {
        const progressContainer = document.getElementById('trainingProgress');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');

        progressContainer.style.display = 'block';
        progressText.textContent = result.message || 'Training in progress...';

        // Simulate progress (in real implementation, this would be updated via WebSocket)
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress >= 100) {
                progress = 100;
                clearInterval(interval);
                progressText.textContent = 'Training completed!';
                this.showToast('Model training completed!', 'success');
            }
            progressFill.style.width = `${progress}%`;
        }, 1000);
    }

    showLoading(text = 'Loading...') {
        const overlay = document.getElementById('loadingOverlay');
        const loadingText = document.getElementById('loadingText');
        loadingText.textContent = text;
        overlay.style.display = 'flex';
    }

    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        overlay.style.display = 'none';
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;

        container.appendChild(toast);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 5000);
    }
}

// Global functions for HTML onclick handlers
function switchSection(sectionName) {
    if (window.app) {
        window.app.switchSection(sectionName);
    }
}

function clearFile() {
    if (window.app) {
        window.app.clearFile();
    }
}

function analyzeFile() {
    if (window.app) {
        window.app.analyzeFile();
    }
}

function startSimpleTraining() {
    if (window.app) {
        window.app.startSimpleTraining();
    }
}

function startProTraining() {
    if (window.app) {
        window.app.startProTraining();
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ExoOperatorApp();
});

// Add some additional CSS for the new elements
const additionalCSS = `
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.875rem;
    }
    
    .data-table th,
    .data-table td {
        padding: 0.5rem;
        text-align: left;
        border-bottom: 1px solid var(--gray-200);
    }
    
    .data-table th {
        background: var(--gray-50);
        font-weight: 600;
        color: var(--gray-700);
    }
    
    .summary-item {
        margin-bottom: 0.5rem;
    }
    
    .probability-list {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    
    .probability-item {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem;
        background: var(--gray-50);
        border-radius: 0.375rem;
    }
    
    .analysis-summary {
        margin-bottom: 2rem;
    }
    
    .summary-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .stat-item {
        display: flex;
        justify-content: space-between;
        padding: 1rem;
        background: var(--gray-50);
        border-radius: 0.5rem;
    }
    
    .stat-label {
        font-weight: 500;
        color: var(--gray-700);
    }
    
    .stat-value {
        font-weight: 600;
        color: var(--primary-color);
    }
    
    .class-distribution {
        margin-bottom: 2rem;
    }
    
    .distribution-chart {
        margin-top: 1rem;
    }
    
    .distribution-item {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.75rem;
    }
    
    .distribution-label {
        min-width: 120px;
        font-weight: 500;
        color: var(--gray-700);
    }
    
    .distribution-bar {
        flex: 1;
        height: 20px;
        background: var(--gray-200);
        border-radius: 10px;
        overflow: hidden;
    }
    
    .distribution-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        transition: width 0.3s ease;
    }
    
    .distribution-value {
        min-width: 80px;
        text-align: right;
        font-weight: 600;
        color: var(--gray-700);
    }
    
    .results-table-container {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid var(--gray-200);
        border-radius: 0.5rem;
    }
    
    .results-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.875rem;
    }
    
    .results-table th,
    .results-table td {
        padding: 0.75rem;
        text-align: left;
        border-bottom: 1px solid var(--gray-200);
    }
    
    .results-table th {
        background: var(--gray-50);
        font-weight: 600;
        color: var(--gray-700);
        position: sticky;
        top: 0;
    }
    
    .results-table tbody tr:hover {
        background: var(--gray-50);
    }
`;

// Inject additional CSS
const style = document.createElement('style');
style.textContent = additionalCSS;
document.head.appendChild(style);
