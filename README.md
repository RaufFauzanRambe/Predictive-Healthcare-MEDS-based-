# Predictive Healthcare MEDS

A comprehensive machine learning framework for predictive healthcare analytics using the Medical Event Data Standard (MEDS).

## Overview

This project provides an end-to-end pipeline for building predictive models on healthcare event data. It supports the MEDS format for standardized medical event representation and includes implementations of various deep learning architectures for temporal prediction tasks such as readmission risk, disease progression, and clinical outcome prediction.

## Features

- **Data Pipeline**: Complete ingestion, cleaning, and MEDS format conversion with support for various healthcare data sources
- **Feature Engineering**: Event-based, temporal, and patient embedding features for rich representation learning
- **Models**: 
  - Baseline Logistic Regression for interpretable predictions
  - Transformer-based sequence model for capturing long-range dependencies
  - Temporal Convolutional Network (TCN) for efficient sequence modeling
- **Evaluation**: Comprehensive metrics, calibration analysis, and model explainability using SHAP and attention visualization
- **Deployment**: FastAPI-based inference service with Docker support for scalable production deployment

## Project Structure

```
predictive-healthcare-meds/
├── configs/              # Configuration files
│   ├── config.yaml       # Main configuration
│   ├── model_config.yaml # Model-specific parameters
│   └── data_config.yaml  # Data processing settings
├── data/                 # Data storage
│   ├── raw/              # Raw input data
│   ├── processed/        # Processed intermediate data
│   ├── meds_format/      # MEDS-formatted data
│   └── external/         # External reference data
├── src/                  # Source code
│   ├── data_pipeline/    # Data processing modules
│   ├── features/         # Feature engineering
│   ├── models/           # Model implementations
│   ├── evaluation/       # Evaluation utilities
│   └── utils/            # Helper utilities
├── notebooks/            # Jupyter notebooks for exploration
├── experiments/          # Experiment tracking and results
├── tests/                # Unit tests
├── scripts/              # Executable scripts
├── docs/                 # Documentation
├── results/              # Output storage
└── deployment/           # Deployment configurations
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip or poetry package manager
- (Optional) CUDA-capable GPU for training deep learning models

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/predictive-healthcare-meds.git
cd predictive-healthcare-meds

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or using poetry
poetry install
```

## Quick Start

### 1. Data Preparation

Place your raw healthcare data in the `data/raw/` directory. The pipeline supports CSV, Parquet, and FHIR formats.

```bash
# Preprocess raw data into MEDS format
python scripts/preprocess.py --config configs/data_config.yaml
```

### 2. Training

```bash
# Train baseline logistic regression model
python scripts/train.py --config configs/config.yaml --model baseline

# Train transformer model
python scripts/train.py --config configs/config.yaml --model transformer

# Train TCN model
python scripts/train.py --config configs/config.yaml --model tcn
```

### 3. Evaluation

```bash
# Evaluate trained model with comprehensive metrics
python scripts/evaluate.py --model-path experiments/exp_001_baseline/model.pkl

# Generate explainability reports
python scripts/evaluate.py --model-path experiments/exp_001_baseline/model.pkl --explain
```

### 4. Deployment

```bash
# Build and run Docker container
cd deployment/docker
docker build -t healthcare-predictor .
docker run -p 8000:8000 healthcare-predictor

# Or run FastAPI directly
cd deployment/api
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Configuration

Configuration files are located in `configs/`:

- `config.yaml`: Main pipeline configuration (paths, logging, experiment settings)
- `model_config.yaml`: Model-specific hyperparameters
- `data_config.yaml`: Data processing and feature engineering settings

## MEDS Format

This project uses the Medical Event Data Standard (MEDS) for representing patient medical histories. MEDS provides a standardized schema for:

- **Patient demographics**: Age, gender, race, ethnicity
- **Clinical events**: Diagnoses, procedures, medications, lab results
- **Temporal information**: Timestamps and duration of events
- **Clinical concepts**: Standardized code mappings (ICD, CPT, LOINC, RxNorm)

See `docs/meds_schema_explanation.md` for detailed schema documentation.

## Model Architecture

### Baseline Logistic Regression
A simple, interpretable baseline using aggregated features and L2 regularization. Suitable for quick prototyping and benchmarking.

### Transformer Model
A transformer-based architecture for sequence modeling:
- Multi-head self-attention for capturing event relationships
- Positional encoding for temporal information
- Layer normalization and dropout for regularization

### Temporal Convolutional Network (TCN)
An efficient convolutional architecture:
- Dilated causal convolutions for receptive field expansion
- Residual connections for gradient flow
- Suitable for long sequences with lower computational cost

## Evaluation Metrics

The framework provides comprehensive evaluation:

- **Classification Metrics**: AUROC, AUPRC, F1, precision, recall, accuracy
- **Calibration**: Calibration curves, Brier score, expected calibration error
- **Explainability**: SHAP values, attention visualization, feature importance

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_models.py -v
```

## Documentation

- [Architecture Overview](docs/architecture.md) - System architecture and design decisions
- [MEDS Schema Explanation](docs/meds_schema_explanation.md) - Detailed MEDS format documentation
- [Dataset Description](docs/dataset_description.md) - Expected data format and examples
- [Model Card](docs/model_card.md) - Model details, limitations, and intended use

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

Please ensure all tests pass and code follows the project style guidelines.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MEDS format specification contributors
- Healthcare ML community
- Open-source healthcare AI projects

## Citation

If you use this code in your research, please cite:

```bibtex
@software{predictive_healthcare_meds,
  title = {Predictive Healthcare MEDS: A Framework for Medical Event Prediction},
  author = {Ra'uf Fauzan Rambe},
  year = {2024},
  url = {https://github.com/yourusername/predictive-healthcare-meds}
}
```

## Contact

For questions and support, please open an issue on GitHub or contact the maintainers.

## Disclaimer

This software is provided for research and educational purposes only. It is not intended for clinical use or medical decision-making without appropriate validation and regulatory approval.
