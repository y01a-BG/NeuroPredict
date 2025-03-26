FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Copy everything first so we have all files including the models
COPY api api
COPY processed_data processed_data
COPY EEG EEG
COPY requirements.txt requirements.txt
COPY models models



# For debugging - verify files exist
RUN ls -la && ls -la models/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install TensorFlow and Keras explicitly
#RUN pip install --no-cache-dir tensorflow-cpu==2.10.0

# Set the Python path to include the project root
#ENV PYTHONPATH=/app

# Expose port based on the PORT environment variable, defaulting to 8080
#ENV PORT=8080
#EXPOSE ${PORT}

# Run the application using the PORT environment variable
#CMD ["uvicorn", "api.app_y01a:app", "--host", "0.0.0.0", "--port", "8080"]
CMD uvicorn api.app_y01a:app --host 0.0.0.0 --port $PORT
