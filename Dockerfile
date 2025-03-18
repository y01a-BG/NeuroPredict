FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code
COPY ./api ./api

# Expose port based on the PORT environment variable, defaulting to 8080
ENV PORT=8080
EXPOSE ${PORT}

# Run the application using the PORT environment variable
CMD uvicorn api.app:app --host 0.0.0.0 --port ${PORT} 