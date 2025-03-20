
import os
import pandas as pd
import numpy as np

from encoding_03 import encoder_LSTM
import tensorflow as tf
from keras.models import load_model

input_path = "./processed_data"
X_pred_file_name = "random_test_samples.csv"
X_pred_file_path = os.path.join(input_path, X_pred_file_name)

X_pred = pd.read_csv(X_pred_file_path)

def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    #if X_pred is None:
        ##########################
        ########## ask FEDERICO     #####
        ##########################


    model = load_model("models/LSTMmodel.h5")
    assert model is not None

    X_processed = encoder_LSTM(X_pred)
    y_pred_probs = model.predict(X_processed)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)

    predictions = []
    for label in y_pred_labels:
        if label == 0:
            prediction_text = "Your EEG is predicted to be tumor-induced seizure EEG."
            predictions.append(prediction_text)
        elif label == 1:
            prediction_text = "Your EEG is predicted to be tumor baseline EEG."
            predictions.append(prediction_text)
        else:
            prediction_text = "Your EEG is predicted to be healthy baseline EEG."
            predictions.append(prediction_text)

    print(predictions)
    return {"predictions": predictions}





pred(X_pred)
