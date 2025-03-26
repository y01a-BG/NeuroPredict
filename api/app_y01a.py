from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import os
import io
import joblib
import pandas as pd
import numpy as np

from sktime.datatypes._panel._convert import from_2d_array_to_nested
from keras.models import load_model
import pickle
from xgboost import XGBClassifier
from EEG.Hybrid_LSTM_BinaryRocket import hybrid_prediction

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


################################################################################
#####################   Always awaiable ########################################
##############################################################################

# load the prediction data
data_pred = pd.read_csv("./processed_data/prediction_data.csv")
X_pred = data_pred.drop(columns = 'y')
#y_pred_origin = data_pred.y

# Load the scalers needed for some models
xgb_scaler = pickle.load(open(f"./models/scaler_xgb.pkl", "rb"))


# Load the models once when the app starts
xgb = XGBClassifier()
xgb.load_model("./models/model_xgb.json")
rocket = joblib.load("./models/model_rocket.pkl")

lstm = load_model("models/model_lstm.h5")
binary_rocket = joblib.load("./models/binary_rocket.pkl")

################################################################################
####################             NEW IDEA          ############################
################################################################################


@app.get("/predict")
async def predict(model: str): # model
    """
    Root endpoint that accepts a CSV file, preprocesses it,
    runs predictions using the LSTM model, and returns the results
    """
    #try:

    if model == 'LSTM':
            X_pred_enc = np.expand_dims(X_pred, axis = 2)  # to be able to enter the LSTM model
            y_pred_probs = lstm.predict(X_pred_enc)
            y_pred = np.argmax(y_pred_probs, axis=1)
    elif model == 'XGBoost':
            X_pred_enc = xgb_scaler.transform(X_pred)
            y_pred = xgb.predict(X_pred_enc)
    elif model == 'Rocket':
            X_pred_enc = from_2d_array_to_nested(X_pred,
                                                index=None, columns=None, time_index=None,
                                                cells_as_numpy=False
                         )
            y_pred = rocket.predict(X_pred_enc)
    elif model == 'Hybrid':
            y_pred = hybrid_prediction(X_pred,lstm, binary_rocket)


        # Map prediction labels to text
    prediction_texts = []
    for label in y_pred:
            if label == 0:
                prediction_texts.append("Your EEG is predicted to be tumor-induced seizure EEG.")
            elif label == 1:
                prediction_texts.append("Your EEG is predicted to be tumor baseline EEG.")
            else:
                prediction_texts.append("Your EEG is predicted to be healthy baseline EEG.")

    # Return predictions along with original row count and filename
    return { "predictions": prediction_texts}

    #except Exception as e:
    #   return {"error": str(e)}









@app.get("/hello/{name}")
async def say_hello(name: str):
    """
    Endpoint that says hello to a specific name
    """
    return {"message": f"Hello, {name}!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)



################################################################################
####################   OLD IDEA - Fabian worked on this ########################
################################################################################


# @app.post("/")
# async def process_csv(csv_file: UploadFile = File(...)): # I should get user model
#     """
#     Root endpoint that accepts a CSV file, preprocesses it,
#     runs predictions using the LSTM model, and returns the results
#     """
#     try:
#         # Read the CSV file content
#         contents = await csv_file.read()

#         # Convert to pandas DataFrame
#         df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

#         # Preprocess data using the encoder_LSTM function
#         processed_data = encoder_LSTM(df)

#         # Run prediction using the model
#         prediction_probs = model.predict(processed_data)
#         prediction_labels = np.argmax(prediction_probs, axis=1)

#         # Map prediction labels to text
#         prediction_texts = []
#         for label in prediction_labels:
#             if label == 0:
#                 prediction_texts.append("Your EEG is predicted to be tumor-induced seizure EEG.")
#             elif label == 1:
#                 prediction_texts.append("Your EEG is predicted to be tumor baseline EEG.")
#             else:
#                 prediction_texts.append("Your EEG is predicted to be healthy baseline EEG.")

#         # Return predictions along with original row count and filename
#         return {
#             "rows": len(df),
#             "filename": csv_file.filename,
#             "predictions": prediction_texts,
#             "prediction_labels": prediction_labels.tolist()
#         }
#     except Exception as e:
#         return {"error": str(e)}
