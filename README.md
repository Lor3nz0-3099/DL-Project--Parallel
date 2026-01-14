# Photovoltaic Forecasting with Temporal Fusion Transformer (TFT)

A comprehensive deep learning project for time series forecasting of photovoltaic (PV) power generation using the state-of-the-art Temporal Fusion Transformer model. This project implements advanced techniques for renewable energy forecasting with full model interpretability and robust validation.

## Features

- **24-hour ahead PV power forecasting** using weather and historical data
- **Temporal cross-validation** with 5-fold validation strategy
- **Automated hyperparameter tuning** with Optuna (TPE sampler + MedianPruner)
- **Advanced model interpretability** through attention weights and feature importance
- **Masked MAPE** implementation to correct nighttime zero-bias
- **Smart execution modes** with result caching and skip functionality
- **Comprehensive evaluation** with multiple metrics (MAE, RMSE, R^2, Masked MAPE)
- **PyTorch Lightning integration** for scalable training with checkpoints
- **GPU acceleration** with CUDA support and mixed precision training

## Project Structure

```
DL_project/
├── data/
│   └── raw/                    # Raw datasets
│       ├── pv_dataset - 07-10--06-11.csv
│       ├── pv_dataset - 07-11--06-12.csv
│       ├── wx_dataset - 07-10--06-11.csv
│       └── wx_dataset - 07-11--06-12.csv
├── src/
│   ├── PV_Forecasting_TFT_Parallel.ipynb  # Main notebook
│   ├── checkpoints/            # Lightning model checkpoints
│   ├── lightning_logs/         # TensorBoard training logs
│   └── results/               # Complete analysis results
│       └── run_corretta/      # Best run results
│           ├── config.json    # Run configuration
│           ├── models/        # Trained TFT models (.ckpt)
│           ├── figures/       # Visualization outputs
│           ├── optuna_study.pkl  # Hyperparameter optimization
│           ├── cv_results.pkl # Cross-validation results
│           └── *.json         # Analysis summaries
├── requirements.txt           # Python dependencies
└── README.md
```

## Dataset

The project uses two types of datasets:
- **PV datasets**: Historical photovoltaic power generation data
- **Weather datasets**: Meteorological data including temperature, humidity, solar irradiance, etc.

Data spans multiple periods from July 2010 to June 2012, providing comprehensive seasonal coverage for robust model training.

## Pipeline Overview

### 1. Data Processing
- **Automatic CSV detection** and multi-file loading
- **Data quality analysis** with missing value handling
- **Physics-based feature scaling** (solar irradiance, weather variables)
- **Temporal continuity checks** and gap detection

### 2. Advanced Feature Engineering
- **Cyclic temporal features** (hour, day, month, DOW) with sin/cos encoding
- **Weather variable transformations** (wind direction, humidity, cloud cover)
- **Solar irradiance normalization** (Dhi, Dni, Ghi)
- **Time series stationarity** and trend analysis

### 3. Hyperparameter Optimization
- **Optuna study** with TPE sampler for intelligent search
- **MedianPruner** for early trial termination
- **Multi-objective optimization** targeting MAE with cross-validation
- **Search space covering** architecture, regularization, and training parameters

### 4. Model Training & Validation
- **5-fold temporal cross-validation** preserving time dependencies
- **PyTorch Lightning** integration with automatic checkpointing
- **Early stopping** with patience-based validation monitoring
- **Mixed precision training** for memory efficiency

### 5. Comprehensive Evaluation
- **Masked MAPE** for daylight-only evaluation (6:00-19:00)
- **Multi-metric assessment**: MAE, RMSE, R² 
- **Quantile regression** for uncertainty estimation
- **Residual analysis** and error distribution visualization

### 6. Model Interpretability
- **Attention weight analysis** for temporal pattern understanding
- **Variable importance ranking** through integrated gradients
- **Prediction examples** with attention visualization
- **Feature contribution analysis** for model explainability

## Technical Stack

- **Deep Learning Framework**: PyTorch 2.7+, PyTorch Lightning 2.6+
- **Time Series Forecasting**: PyTorch Forecasting
- **Hyperparameter Optimization**: Optuna 4.6+ (TPE sampler, MedianPruner)
- **Data Processing**: Pandas 2.3+, NumPy 2.3+, Scikit-learn
- **Visualization**: Matplotlib, Seaborn (darkgrid style)
- **Model Persistence**: Pickle, JSON for configuration management
- **Development**: Jupyter Notebook with rich logging and progress tracking
- **Hardware**: CUDA-enabled GPU support (tested on RTX 4070)

## Configuration

The notebook supports 3 flexible execution modes with intelligent caching:
- hyperparameter tuning and final best model training are performed
- hyperparameter tuning is skipped, but final best model training is performed
- hyperparameter tuning and final best model training are both skipped (using a referece fold for results)

```python
# Execution Control Flags
RUN_TUNING_AND_CV = False    # Set to True for hyperparameter optimization
RUN_FINAL_TRAINING = False   # Set to True for final model training 
REFERENCE_RUN_FOLDER = "run_corretta"  # Use existing results from this folder

# Main Configuration
CONFIG = {
    # TFT Architecture
    'MAX_ENCODER_LENGTH': 168,      # 1 week context (hours)
    'MAX_PREDICTION_LENGTH': 24,    # 24-hour forecast horizon
    
    # Cross-Validation
    'N_FOLDS': 5,                   # Temporal CV folds
    'VAL_RATIO': 0.2,              # 20% validation per fold
    
    # Hyperparameter Tuning
    'N_TRIALS': 5,                  # Optuna trials
    'OPTUNA_TIMEOUT_HOURS': 6,      # Maximum optimization time
    
    # MAPE Masking (Critical for PV forecasting)
    'DAYLIGHT_START_HOUR': 6,       # Start hour for MAPE calculation
    'DAYLIGHT_END_HOUR': 19,        # End hour for MAPE calculation
    'EPS_PV': 5.0,                  # Min power threshold (kW)
    
    # Training Parameters
    'MAX_EPOCHS_TUNING': 30,        # Epochs during hyperparameter search
    'MAX_EPOCHS_FINAL': 150,        # Epochs for final training
    'EARLY_STOPPING_PATIENCE': 15,  # Early stopping patience
    
    'SEED': 42,                     # Reproducibility seed
    'ACCELERATOR': 'auto'           # 'auto', 'gpu', 'cpu'
}
```

## Model Architecture

**Temporal Fusion Transformer (TFT)** components:
- **Variable Selection Networks**: Automatic feature selection
- **Gated Residual Networks**: Non-linear processing
- **Multi-Head Attention**: Temporal relationship modeling
- **Quantile Regression**: Uncertainty quantification

### Input Features
- **Time-varying unknown**: PV power, weather variables
- **Time-varying known**: Hour, day, month, weekday
- **Static categorical**: System identifiers (if applicable)

## Performance Metrics

- **MAE (Mean Absolute Error)**: Primary optimization metric
- **RMSE (Root Mean Square Error)**: For variance assessment
- **R² (R-squared)**: Coefficient of determination for model fit quality
- **Masked MAPE**: Corrected for nighttime zero values
- **Quantile Losses**: For uncertainty estimation

## Key Innovations

1. **Masked MAPE Implementation**: Addresses mathematical issues with traditional MAPE when actual values are zero (nighttime PV generation), focusing evaluation on daylight hours (6:00-19:00) with power threshold filtering

2. **Temporal Cross-Validation Strategy**: 5-fold progressive validation respecting time dependencies, with 20% validation ratio per fold ensuring robust performance estimation

3. **Smart Execution Pipeline**: 
   - Intelligent result caching and reuse system
   - Skip modes for completed processing stages
   - Reference folder system for experiment reproducibility
   - Comprehensive logging and progress tracking

4. **Advanced Model Interpretability**:
   - Attention weight visualization for temporal patterns
   - Feature importance ranking through gradient analysis  
   - Prediction example generation with attention maps
   - Model decision transparency for renewable energy insights

5. **PyTorch 2.x Compatibility**: Automatic handling of `weights_only` parameter for checkpoint loading compatibility across PyTorch versions

6. **Robust Hyperparameter Optimization**:
   - MedianPruner for efficient trial termination
   - TPE sampler for intelligent parameter space exploration
   - Multi-fold validation during optimization to prevent overfitting

## Usage

### Quick Start

1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**: 
   - Place PV datasets in `data/raw/` (format: `pv_dataset - *.csv`)
   - Place weather datasets in `data/raw/` (format: `wx_dataset - *.csv`)
   - Automatic CSV detection will find and load all matching files

3. **Configuration**:
   ```python
    # For full pipeline execution
    RUN_TUNING_AND_CV = True
    RUN_FINAL_TRAINING = True
    
    # For using existing results
    RUN_TUNING_AND_CV = False
    RUN_FINAL_TRAINING = False
    REFERENCE_RUN_FOLDER = "run_corretta"

    # For skipping optimization and executing final training
    RUN_TUNING_AND_CV = False
    RUN_FINAL_TRAINING = True
    REFERENCE_RUN_FOLDER = "run_corretta"

    #Unsupported configuration (not so much useful)
    RUN_TUNING_AND_CV = True
    RUN_FINAL_TRAINING = False
    REFERENCE_RUN_FOLDER = "run_corretta"
   ```


4. **Execution Modes**:
   - **Full Pipeline**: Both flags `True` - complete hyperparameter optimization and training
   - **Training Only**: `RUN_TUNING_AND_CV=False, RUN_FINAL_TRAINING=True` - use existing hyperparameters
   - **Analysis Only**: Both flags `False` - load existing results for analysis and interpretability

5. **Results Access**:
   - Check `src/results/[run_id]/` for complete outputs
   - Models saved in `models/best_tft_fold_*.ckpt`
   - Figures available in `figures/` subdirectory
   - JSON files contain structured analysis results

## Expected Outputs

### Results Directory Structure (`results/run_*/`):
- **config.json**: Complete run configuration and parameters
- **models/**: Trained TFT models (.ckpt files) for each CV fold
- **figures/**: All visualization outputs (data overview, interpretability plots)
- **optuna_study.pkl**: Complete hyperparameter optimization history
- **cv_results.pkl**: Cross-validation results and fold performance
- **comprehensive_report.json**: Summary metrics and analysis
- **optimization_analysis.json**: Hyperparameter search analysis
- **lightning_logs/**: TensorBoard-compatible training logs

## Model Performance

### Achieved Results:
- **Best MAE**: 3.89 kW on 5-fold cross-validation
- **Optimization Efficiency**: 60% trial completion rate with MedianPruner
- **Training Stability**: 94 epochs optimal with early stopping
- **Validation Improvement**: 53.55% total loss reduction during training

### Model Capabilities:
- **24-hour horizon forecasting** with sub-4 kW MAE
- **Weather pattern integration** through multi-head attention
- **Seasonal adaptation** via temporal feature encoding
- **Uncertainty quantification** through quantile regression
- **Interpretable predictions** with attention weight analysis

## Notes

- The project implements advanced time series forecasting techniques specifically designed for renewable energy applications
- Model checkpoints are automatically saved during training for reproducibility
- The interpretability analysis helps understand the physical relationships learned by the model
- Results are saved in structured formats for easy analysis and comparison

## Future Developments

### Short-term Improvements
- **Expanded Hyperparameter Search**: Increase from 5 to 50+ Optuna trials for better optimization
- **Inceease the Dataset Size**: From 17000 sample to at least 100k for effective use of TFT's capabilities. 
- **Enhanced Weather Features**: Integrate satellite imagery, cloud movement patterns, and weather forecasts

### Advanced Research Directions
- **Ensemble Methods**: Combine TFT with other time series models (LSTM, Prophet, N-BEATS)
- **Transfer Learning**: Pre-train on large-scale solar datasets and fine-tune for specific installations
- **Probabilistic Forecasting**: Enhanced uncertainty quantification with distributional regression
- **Explainable AI**: Advanced SHAP/LIME integration for stakeholder-friendly model explanations

### Technical Enhancements
- **Distributed Training**: Multi-GPU training with data parallelism for larger datasets

### Data Science Extensions
- **Synthetic Data**: GAN-based data augmentation for rare weather conditions
- **Multi-modal Learning**: Integration of weather images, satellite data, and IoT sensors

---

*This project demonstrates the application of state-of-the-art deep learning techniques to renewable energy forecasting, combining technical rigor with practical applicability.*