# NeuroPredict: EEG-Based Tumor and Seizure Classification

This repository is a **monorepo** comprising both the model backend and the web-based frontend for NeuroPredict, a tool for classifying EEG signals to predict tumor and seizure activity. It combines all relevant information, dependencies, and instructions from both the FastAPI/Streamlit model backend (`y01a-BG/NeuroPredict`) and the standalone Streamlit app (`fglaw/NeuroPredictFrontend`). This README provides accurate, actionable steps to seamlessly fork, set up, and run the entire project.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objective](#objective)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Functionality](#functionality)
- [Running the Application](#running-the-application)
- [Dependencies & Environment](#dependencies--environment)
  - [Backend (API)](#backend-api)
  - [Frontend (Streamlit)](#frontend-streamlit)
  - [Development/Notebook Environment](#developmentnotebook-environment)
- [Docker Usage](#docker-usage)
- [How to Use](#how-to-use)
- [Future Enhancements](#future-enhancements)
- [References](#references)

---

## Project Overview

NeuroPredict is a machine learning platform designed to assist in medical diagnosis by classifying EEG (electroencephalogram) data into clinically relevant categories: healthy baseline, tumor baseline, and tumor-induced seizure. The platform consists of:

- **Backend:** FastAPI service for prediction and data processing.
- **Frontend:** Streamlit app for data upload, visualization, and prediction display.
- **Jupyter Notebooks:** For model development, data exploration, and reproducible research.

---

## Objective

- Classify EEG signals into:
  - **Healthy baseline**
  - **Tumor baseline**
  - **Tumor-induced seizure**

---

## Dataset

- Source: [Epileptic Seizure Recognition (Kaggle)](https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition/data)
- **Description:**
  - 500 subjects, each with 4097 EEG data points (23.5 seconds)
  - Data is reshaped into 23 segments per individual (178 data points each)
  - Labels:
    - 1: Seizure activity
    - 2: Tumor-affected area
    - 3: Healthy area adjacent to tumor
    - 4: Healthy baseline (tumor patient)
    - 5: Eyes open baseline (healthy patient)

---

## Project Structure

```
project-root/
├── api/                # FastAPI backend
│   ├── app.py
│   └── fast.py
├── frontend/           # Streamlit frontend
│   └── app.py
├── models/
│   └── LSTMmodel.h5    # Pre-trained model
├── packagename/
│   └── encoding_03.py  # Data preprocessing
├── notebooks/          # Data exploration & dev
├── requirements.txt    # Main dependencies
├── requerments_dev.txt # Dev & notebook dependencies
├── Dockerfile
├── Makefile
└── README.md
```

---

## Functionality

### Backend (FastAPI)

- **/upload** endpoint: Accepts CSV uploads of EEG data.
- **Preprocessing:** Uses `encoder_LSTM` for input transformation.
- **Prediction:** Loads and uses `LSTMmodel.h5` for classification.
- **Response:** Returns structured predictions (healthy, tumor, seizure).

### Frontend (Streamlit)

- **CSV Upload:** Users can upload EEG CSV files.
- **Visualization:** Shows time-series plots for the first 6 rows.
- **Prediction Display:** Results are color-coded:
  - Green: Healthy baseline
  - Yellow: Tumor baseline
  - Red: Tumor-induced seizure

---

## Running the Application

### Prerequisites

- Python 3.8, 3.9, 3.10, or 3.11 (recommended: **3.10** for maximal compatibility)
- pip (latest version highly recommended)
- [Docker](https://docs.docker.com/get-docker/) (optional, for containerized runs)

### Backend (API)

```bash
cd api
pip install -r ../requirements.txt
uvicorn app:app --reload
```
API will be available at `http://localhost:8000`.

### Frontend (Streamlit)

```bash
cd frontend
pip install -r ../requirements.txt
streamlit run app.py
```
Streamlit app will be available at `http://localhost:8501`.

### Both Services (Docker Compose)

```bash
docker-compose up --build
```
- FastAPI: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

> **Note:** The public demo may be available at: https://neuropredictor.streamlit.app/

---

## Dependencies & Environment

### Main Application (`requirements.txt`)

These are the **minimum required versions** for production (API + Streamlit):

```text
# API
fastapi==0.110.0
uvicorn==0.27.1
starlette==0.36.3
pydantic==2.10.6
pandas==2.0.3
python-multipart==0.0.9
joblib==1.4.2
sktime==0.36.0
xgboost==3.0.0
numba==0.61.0

# Streamlit Frontend
streamlit==1.32.0
requests==2.31.0
matplotlib==3.8.2
plotly==5.9.0
seaborn==0.13.1
numpy==1.24.1
setuptools==69.0.3

# ML/DL
tensorflow-cpu==2.10.0
keras==2.10.0
```

**Important Version Notes:**
- `pandas==2.0.3` (API) and `pandas==2.2.0` (frontend) are both referenced in the two repos; `2.0.3` is safest for backend, `2.2.0` works for frontend but ensure compatibility with other ML libs.
- `numpy==1.24.1` (for notebooks, dev, and API); some notebook cells require `numpy<2.2`.
- `tensorflow-cpu==2.10.0` and `keras==2.10.0` are standard for cross-platform compatibility.
- `sktime`, `xgboost`, and `numba` are included for advanced time-series modeling and ML features.

### Development/Notebook Environment (`requerments_dev.txt`)

For running and reproducing Jupyter notebooks or contributing to model development:

```text
wheel
nbresult
colorama
ipdb
ipykernel
yapf
matplotlib
pygeohash
pytest
seaborn
xgboost
numpy==1.24.1
pandas==1.5.3
scipy==1.10.0
scikit-learn==1.3.1
google-cloud-bigquery
google-cloud-storage==2.14.0
google-api-core==2.8.2
googleapis-common-protos==1.56.4
protobuf==3.19.6
h5py==3.10.0
db-dtypes
pyarrow
mlflow==2.1.1
prefect==2.19.2
python-dotenv
psycopg2-binary
fastapi==0.108.0
pytz
uvicorn
httpx<0.28
pytest-asyncio
# For Mac (M1/M2): tensorflow-macos==2.10.0
# For Mac (Intel): tensorflow==2.10.0
# For Linux/Windows: tensorflow==2.10.0
```

> **Note:** For Mac with Apple Silicon, use `tensorflow-macos`; for all other platforms, use regular `tensorflow==2.10.0`.

### Hardware/OS Notes

- Works best on Linux or MacOS.
- For GPU support, additional CUDA libraries are required (see TensorFlow docs).
- For headless servers, use the `cpu` version of TensorFlow.

---

## Docker Usage

A sample `Dockerfile` is included for production deployment:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY api api
COPY processed_data processed_data
COPY EEG EEG
COPY requirements.txt requirements.txt
COPY models models

RUN pip install --no-cache-dir -r requirements.txt

# Expose port as needed (e.g., 8000 for API)
```

For cloud deployment (e.g., Google Cloud Run), see the included `Makefile`.

---

## How to Use

1. **Clone the repository:**
   ```bash
   git clone https://github.com/y01a-BG/NeuroPredict.git
   cd NeuroPredict
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Set up development environment:**
   ```bash
   pip install -r requerments_dev.txt
   ```

4. **Run backend (API):**
   ```bash
   cd api
   uvicorn app:app --reload
   # Accessible at http://localhost:8000
   ```

5. **Run frontend (Streamlit):**
   ```bash
   cd ../frontend
   streamlit run app.py
   # Accessible at http://localhost:8501
   ```

6. **Use Docker Compose for both:**
   ```bash
   docker-compose up --build
   ```

7. **Upload an EEG CSV file via the Streamlit app.**
   The frontend will display visualizations and color-coded predictions.

---

## Future Enhancements

- Improved model accuracy & feature engineering
- More detailed EEG visualizations
- Real-time EEG monitoring
- Explainability and interpretability features
- Expanded neurological condition detection

---

## References

- Andrzejak et al., 2001. Indications of nonlinear deterministic and finite dimensional structures in brain activity.
- Kode et al., 2024 (project contributors)
- [Epileptic Seizure Recognition Dataset – Kaggle](https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition/data)

---

**NeuroPredict** is dedicated to advancing AI-assisted medical diagnostics with a focus on explainability, ethics, and data privacy.

---

### For contributors

- Please use branches for PRs.
- For any questions about environment setup, see the [Dependencies & Environment](#dependencies--environment) section.
- If you encounter installation errors (especially with pandas/numpy), ensure your `pip`, `setuptools`, and Python version are up-to-date and match the above specifications.
