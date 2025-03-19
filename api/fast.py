import os
import pandas as pd
import numpy as np

from keras.models import load_model
from packagename.encoding_03 import encoder_LSTM

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


# Load the model once when the app starts (so it doesn't need to be reloaded on every request)
model = load_model("models/LSTMmodel.h5")
assert model is not None


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model.
    """
    if X_pred is None:
        raise ValueError("Input data is required for prediction.")

    # Process data
    X_processed = encoder_LSTM(X_pred)
    y_pred_probs = model.predict(X_processed)
    y_pred_label = np.argmax(y_pred_probs, axis=1)
    return y_pred_label


@app.post("/predict/")
async def make_prediction(input_data):
    """
    Endpoint to accept the prediction input and return the model's prediction.
    """

    # Convert input data to DataFrame
    X_pred = pd.DataFrame(input_data.data)

     # Call the prediction function
    y_pred_label = pred(X_pred)

    # Mapping prediction to labels (you can customize this part as needed)
    if y_pred_label == 0:
        prediction_text = "Your EEG is predicted to be tumor-induced seizure EEG."
    elif y_pred_label == 1:
        prediction_text = "Your EEG is predicted to be tumor baseline EEG."
    else:
        prediction_text = "Your EEG is predicted to be healthy baseline EEG."

    return {"prediction": prediction_text}



@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# Run: `uvicorn api.fast:app --reload`
