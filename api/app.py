from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import pandas as pd
import io
import numpy as np
from keras.models import load_model
from packagename.encoding_03 import encoder_LSTM

# Get port from environment variable for Cloud Run compatibility
PORT = int(os.getenv("PORT", "8000"))

# Create FastAPI instance
app = FastAPI(
    title="Hello World API",
    description="A minimal FastAPI Hello World example",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the model once when the app starts
model = load_model("models/LSTMmodel.h5")

@app.post("/")
async def process_csv(csv_file: UploadFile = File(...)):
    """
    Root endpoint that accepts a CSV file, preprocesses it, 
    runs predictions using the LSTM model, and returns the results
    """
    try:
        # Read the CSV file content
        contents = await csv_file.read()
        
        # Convert to pandas DataFrame
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Preprocess data using the encoder_LSTM function
        processed_data = encoder_LSTM(df)
        
        # Run prediction using the model
        prediction_probs = model.predict(processed_data)
        prediction_labels = np.argmax(prediction_probs, axis=1)
        
        # Map prediction labels to text
        prediction_texts = []
        for label in prediction_labels:
            if label == 0:
                prediction_texts.append("Your EEG is predicted to be tumor-induced seizure EEG.")
            elif label == 1:
                prediction_texts.append("Your EEG is predicted to be tumor baseline EEG.")
            else:
                prediction_texts.append("Your EEG is predicted to be healthy baseline EEG.")
        
        # Return predictions along with original row count and filename
        return {
            "rows": len(df),
            "filename": csv_file.filename,
            "predictions": prediction_texts,
            "prediction_labels": prediction_labels.tolist()
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/hello/{name}")
async def say_hello(name: str):
    """
    Endpoint that says hello to a specific name
    """
    return {"message": f"Hello, {name}!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
