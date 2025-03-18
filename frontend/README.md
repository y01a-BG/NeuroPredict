# Streamlit Frontend

A simple Streamlit frontend application that connects to the FastAPI backend.

## Features

- Interactive "Hello World" interface
- Name input with personalized greeting
- Connection to FastAPI backend endpoints
- Custom styling with CSS

## Running the Application

### Local Development

1. Install the required dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

### Using Docker

1. Build and run using Docker Compose from the root directory:
   ```bash
   docker-compose up --build
   ```

2. Access the Streamlit application at:
   ```
   http://localhost:8501
   ```

## Connecting to the FastAPI Backend

By default, the application connects to the FastAPI backend at `https://hello-world-service-zsuoxbhheq-uc.a.run.app/`. 
You can change this URL in the application UI if your backend is running at a different location.

## Project Structure

- `app.py`: Main Streamlit application
- `Dockerfile`: Docker configuration for the frontend
- `README.md`: Documentation 