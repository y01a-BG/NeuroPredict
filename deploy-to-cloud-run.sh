#!/bin/bash

# Script to deploy NeuroPredict API to Google Cloud Run
echo "Starting deployment of NeuroPredict API to Google Cloud Run..."

# Step 1: Set project ID
PROJECT_ID="lewagon-448710"
echo "Setting project ID to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Step 2: Enable required services
echo "Enabling required Google Cloud services..."
gcloud services enable artifactregistry.googleapis.com run.googleapis.com containerregistry.googleapis.com

# Step 3: Push Docker image to Google Container Registry
echo "Pushing Docker image to Google Container Registry..."
docker push gcr.io/$PROJECT_ID/neuropredict-api:latest

# Step 4: Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy neuropredict-api \
  --image gcr.io/$PROJECT_ID/neuropredict-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --port 8080 \
  --min-instances 0 \
  --max-instances 3

echo "Deployment complete! Your API should be available at the URL provided above."
echo "Test it with: curl [YOUR_CLOUD_RUN_URL]/hello/world" 