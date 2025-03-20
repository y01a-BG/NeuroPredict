# NeuroPredict: EEG-Based Tumor and Seizure Classification

## Project Overview

This project aims to develop a machine learning classification model to predict the presence of tumors and seizure occurrences based on EEG (electroencephalogram) data. By leveraging advanced machine learning and deep learning techniques, the project seeks to create an accurate, ethical, and explainable predictive tool for clinical use.

## Objective
- Classify EEG signals into distinct medical states:
  - Healthy baseline
  - Tumor baseline
  - Tumor-induced seizure

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
├── api/
│   ├── app.py         # FastAPI backend for EEG data processing and prediction
│   └── fast.py        # Additional API functionality
├── frontend/
│   └── app.py         # Streamlit frontend for visualization
├── models/
│   └── LSTMmodel.h5   # Pre-trained LSTM neural network model
├── packagename/
│   └── encoding_03.py # Data preprocessing utilities
└── README.md
```

## Current Functionality

### Backend (FastAPI)

The FastAPI backend provides an API for processing EEG data CSV files and making predictions using a pre-trained LSTM model:

- **CSV Upload Endpoint**: Accepts CSV files containing EEG data
- **Data Preprocessing**: Automatically preprocesses uploaded data using the encoder_LSTM function
- **Model Prediction**: Uses the pre-trained LSTM model to classify EEG signals
- **Response**: Returns prediction results with classifications (healthy baseline, tumor baseline, or tumor-induced seizure)

### Frontend (Streamlit)

The Streamlit frontend provides a user-friendly interface for:

- **CSV File Upload**: Easy upload of EEG data files
- **Data Preview**: Shows a preview of the uploaded data
- **API Integration**: Automatically sends data to the FastAPI backend for processing
- **EEG Signal Visualization**: Displays the first 6 rows of data as time-series plots
- **Prediction Display**: Shows color-coded predictions below each EEG signal:
  - Green: Healthy baseline EEG
  - Yellow: Tumor baseline EEG
  - Red: Tumor-induced seizure EEG

## Running the Application

### Prerequisites

- Python 3.8+
- Required libraries (install using `pip install -r requirements.txt`):
  - fastapi
  - uvicorn
  - streamlit
  - pandas
  - numpy
  - matplotlib
  - keras
  - tensorflow
  - python-multipart
  - requests

### Starting the Backend

```bash
cd api
uvicorn app:app --reload
```

The API will be available at http://127.0.0.1:8000

### Starting the Frontend

```bash
streamlit run frontend/app.py
```

The Streamlit interface will be available at http://127.0.0.1:8501

### Using Docker

If you prefer to use Docker:

```bash
docker-compose up --build
```

This will start both the backend and frontend services.

## How to Use

1. Open the Streamlit frontend at http://127.0.0.1:8501
2. Enter the FastAPI URL (default: http://127.0.0.1:8000)
3. Upload a CSV file containing EEG data
4. View the data preview and wait for automatic processing
5. Examine the EEG signal visualizations and corresponding predictions
6. Each prediction is color-coded to indicate the classification type

## Future Enhancements

- Improve model accuracy with advanced feature engineering
- Add more detailed visualization options for EEG data
- Implement real-time EEG monitoring capabilities
- Add explainability features to help interpret predictions
- Expand classification capabilities to detect more neurological conditions

## References
- Andrzejak et al., 2001. Indications of nonlinear deterministic and finite dimensional structures in brain activity.
- Kode et al., 2024

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
