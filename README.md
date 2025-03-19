# NeuroPredict: EEG-Based Tumor and Seizure Classification

## Project Overview

This project aims to develop a machine learning classification model to predict the presence of tumors and seizure occurrences based on EEG (electroencephalogram) data. By leveraging advanced machine learning and deep learning techniques, the project seeks to create an accurate, ethical, and explainable predictive tool for clinical use.

## Objective
- Classify EEG signals into distinct medical states:
  - Healthy baseline
  - Tumor baseline
  - Tumor seizure
  - Seizure vs. silent stage
  - Healthy vs. tumor 'silent' stage

## Dataset
The dataset used for this project is derived from the [Epileptic Seizure Recognition](https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition/data) dataset provided on Kaggle.

### Data Description:
- 500 subjects
- 4097 EEG data points per individual (23.5 seconds recording)
- Data reshaped into 23 segments per individual (each chunk: 178 data points)
- Labels:
  - 1: Recording of seizure activity
  - 2: EEG from tumor-affected areas
  - 3: EEG from healthy areas adjacent to tumors
  - 4: Healthy baseline (tumor patient)
  - 5: Eyes open baseline (healthy patient)

## Project Structure

```
project/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── exploration.ipynb
│   ├── preprocessing.ipynb
│   ├── feature_engineering.ipynb
│   └── model_training.ipynb
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   └── model.py
├── models/
├── reports/
└── README.md
```

## Key Tasks & Methodology

### 1. Data Exploration & Preprocessing
- Inspect and clean dataset
- Exploratory Data Analysis (EDA)
- Signal processing techniques: Fourier Transform, Wavelet Transform

## Feature Engineering

- Extraction of time-domain, frequency-domain, and statistical features
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Feature selection (Mutual Information, LASSO, SHAP)

## Modeling

### Algorithms
- Logistic Regression
- Support Vector Machines (SVM)
- Decision Trees & Random Forest
- XGBoost
- Deep Learning (CNN, RNN, LSTM)

### Optimization
- Hyperparameter tuning (GridSearchCV, Optuna, Hyperopt)
- Evaluation metrics: accuracy, precision, recall, F1-score, AUC-ROC

## Evaluation & Explainability

- Confusion matrix and classification report analysis
- Model interpretability using SHAP, LIME, Integrated Gradients
- Model calibration and confidence estimation

## Deployment & Monitoring

- Packaging (FastAPI, Docker, Flask)
- Deployment (AWS, GCP, Azure)
- Monitoring (MLflow, Prefect, Airflow)

## Deliverables

- Technical report and executive summary
- Live demo using Streamlit
- Comprehensive final presentation

## References
- Andrzejak et al., 2001. Indications of nonlinear deterministic and finite dimensional structures in brain activity.
- Kode et al., 2024

## Team & Contributions

- Clearly document each team member's contributions to the phases and tasks of the project.

---

**NeuroPredict** is dedicated to advancing medical diagnostics using cutting-edge AI techniques, ensuring explainability, ethics, and compliance with data privacy standards.

## Project Components

### FastAPI Hello World Application

A simple FastAPI Hello World application is included in the `api` directory.

#### Running the FastAPI Application

##### Without Docker

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the FastAPI application:
   ```bash
   cd api
   uvicorn app:app --reload
   ```

3. Access the API at:
   - Main endpoint: http://localhost:8000/
   - Hello endpoint: http://localhost:8000/hello/{name}
   - API documentation: http://localhost:8000/docs

##### With Docker

1. Build and start the Docker container:
   ```bash
   docker-compose up --build api
   ```

2. Access the API at:
   - Main endpoint: http://localhost:8000/
   - Hello endpoint: http://localhost:8000/hello/{name}
   - API documentation: http://localhost:8000/docs

### Streamlit Frontend Application

A simple Streamlit frontend application is included in the `frontend` directory.

#### Running the Streamlit Application

##### Without Docker

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run frontend/app.py
   ```

3. Access the Streamlit app at:
   - http://localhost:8501

##### With Docker

1. Build and start the Docker container:
   ```bash
   docker-compose up --build frontend
   ```

2. Access the Streamlit app at:
   - http://localhost:8501

### Running Both Services Together

1. Build and start both the FastAPI and Streamlit containers:
   ```bash
   docker-compose up --build
   ```

2. Access the applications at:
   - FastAPI: http://localhost:8000
   - Streamlit: http://localhost:8501
