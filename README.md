# Fraud Detection Project

## Course Description

In this course, we will be working on a complete Fraud Detection project to understand how feature engineering, pre processing, model training, evaluation, and production style machine learning pipelines are built in real world systems. Throughout the project, we will process transaction data, build fraud prediction models, automate pre processing workflows, evaluate model performance, and prepare the pipeline for deployment and monitoring.

## Project Overview

This repository contains the code, data, and pipeline logic for a hands-on fraud detection project. The goal is to teach how to convert raw transaction data into features, train and evaluate models, and assemble reproducible pipelines suitable for productionization and monitoring.

## Repository Structure

- `fraud_data.csv` - raw transaction dataset used for the project.
- `processed_features.csv` - example output of the feature pipeline.
- `feature pipeline/feature_pipeline.py` - Python module implementing feature engineering and preprocessing steps.
- `fraud/` - Python virtual environment used for the project.

## Setup

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv fraud
& .\fraud\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. If a `requirements.txt` is not present, install common data science packages:

```powershell
pip install pandas numpy scikit-learn joblib
```

## Usage

- To run the feature engineering pipeline and produce `processed_features.csv`:

```powershell
python "feature pipeline\feature_pipeline.py"
```

- To explore or train models, open the scripts or notebooks in the repo and follow inline instructions.

## Feature Pipeline

The `feature pipeline/feature_pipeline.py` script centralizes data loading, cleaning, feature construction, and output serialization. It is designed so components can be reused inside a training script or wrapped in a production workflow.

## Evaluation and Monitoring

Model evaluation should include common metrics (precision, recall, F1, AUC) and calibration checks. For production, add monitoring for data drift, model performance degradation, and pipeline failures.

## Contributing

Contributions are welcome. Please open issues for bugs or feature requests, and submit pull requests with clear descriptions of changes.

## License

This project is provided for educational purposes. Add a license file if you intend to open-source this work.
