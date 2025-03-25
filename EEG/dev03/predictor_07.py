
import os
import pandas as pd
import numpy as np

from encoding_03 import encoder_LSTM, encoder_XGBoost
import tensorflow as tf
from keras.models import load_model

#****************************************************************#
#****************************************************************#
#****************************************************************#
#   THIS .py CONTAINS trainingpredictions FOR  DIFFERENT ML/DL -models:
#   1. LSTM
#   2. XG Boost
#   3.
#****************************************************************#
#****************************************************************#
#****************************************************************#



#######################################################################
################### Predict LSTM  ###################################
######################################################################


def pred_lstm(X_pred: pd.DataFrame = None) -> dict:
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
    pred_dictionary = {"predictions": predictions}
    print(f'✅ Prediction results for 6 EEG samples are : {pred_dictionary}')
    return pred_dictionary


#######################################################################
################### Predict XGBoost  ###################################
######################################################################


def pred_xgb(X_pred: pd.DataFrame = None) -> dict:
    model = load_model("models/XGBmodel.h5")
    assert model is not None

    X_processed = encoder_XGBoost(X_pred)
    y_pred = model.predict(X_processed)

    predictions = []
    for label in y_pred:
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
    pred_dictionary = {"predictions": predictions}
    print(f'✅ Prediction results for 6 EEG samples are : {pred_dictionary}')
    return pred_dictionary




if __name__ == "__main__":

####### Load 6 random samples  data from processed_data   ####################
######## Valid for all models #######################################

    input_path = "./processed_data"
    X_pred_file_name = "random_test_samples.csv"
    X_pred_file_path = os.path.join(input_path, X_pred_file_name)

    X_pred = pd.read_csv(X_pred_file_path)

    # calling a function
    #pred_dictionary_lstm = pred_lstm(X_pred)
    pred_dictionary_xgb = pred_xgb(X_pred)
