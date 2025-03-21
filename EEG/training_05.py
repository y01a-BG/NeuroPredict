import os
import pandas as pd
import numpy as np

from modeling_04 import initialize_lstm_model, compile_lstm_model, initialize_xgb_model
from encoding_03 import encoder_LSTM, encoder_XGBoost
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras import callbacks
from keras.callbacks import EarlyStopping


from colorama import Fore, Style
from typing import Tuple



#****************************************************************#
#****************************************************************#
#****************************************************************#
#   THIS .py CONTAINS training FOR  DIFFERENT ML/DL -models:
#   1. LSTM
#   2. XG Boost
#   3.
#****************************************************************#
#****************************************************************#
#****************************************************************#



#######################################################################
#################### LSTM model ######################################
######################################################################

def utils_lstm(input_shape = (178,1)):
    # Data encoding from  encoding_03.py
    X_train_enc, y_train_enc = encoder_LSTM(X_train,y_train)
    # Calling the compiled model from modeling_03.py
    model = initialize_lstm_model(input_shape)
    model = compile_lstm_model(model, learning_rate=0.001)

    return X_train_enc, y_train_enc, model



### Define function to TRAIN the model  #################


def train_lstm_model(
        model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=128,
        patience=1,
        validation_data=None, # overrides validation_split
        validation_split=0.2
    ):
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)


    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X,y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=150,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )

    print(f"✅ LSTM Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_loss']), 2)}")

    return model, history




#######################################################################
#################### XGBoost model ######################################
######################################################################


## Training the model (Early Stopping and best model is included in .fit() in XGBoost)

def train_xgb_model(X_train,y_train):

    # Validation data need to be given explicitely: split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    ####Initializing the model from modeling_04
    xgb = initialize_xgb_model()
    best_xgb = xgb.fit(
        X_train,y_train,
        eval_set=[(X_train,y_train), (X_val, y_val)],
        verbose=True
    )
    print(f"✅ XGBoost Model trained")

    return best_xgb




#################################################
######### if runing: python  modelin_04.py ######
#################################################

def training_models(X_train,y_train,model: str):
    """
    This function will apply encoding based on the model choice (LSTM or XGBoost).

    Parameters:
    - X_train: Input training data
    - y_train: Target training data
    - model: A string indicating the model type ('LSTM' or 'XGBoost')

    Returns:
    - Encoded X_train and y_train based on the selected model.
    """

    if model == "LSTM":
        X_train_last, y_train_last, model =  utils_lstm(input_shape = (178,1))
        model_lstm,history = train_lstm_model(model,
                                            X_train_last, y_train_last,
                                            batch_size=128,
                                            patience=1,
                                            validation_data=None, # overrides validation_split
                                            validation_split=0.2
        )
        #model_lstm.save("models/LSTMmodel.h5")
        print("✅ LSTM Model trained on (X_train, y_train)")
        return model_lstm


    elif model == "XGBoost":
        # Encoding the data for XGBoost; this is StandardScaler - output data are np.array
        X_train, y_train = encoder_XGBoost(X_train, y_train)
        xgb = train_xgb_model(X_train, y_train)
        #best_xgb.save("models/XGBoost_model.h5")
        print("✅ XGBoost Model trained on (X_train, y_train)")
        return xgb
    else:
        print(f"Unknown model: {model}. Please choose either 'LSTM' or 'XGBoost'.")



if __name__ == "__main__":


####### Load train  data from processed_data   ####################
#######  this part is valid for all models!!! #####################
######################################################################

    input_path = "./processed_data"
    X_train_file_name = "X_train.csv"
    y_train_file_name = "y_train.csv"
    X_train_file_path = os.path.join(input_path, X_train_file_name)
    y_train_file_path = os.path.join(input_path, y_train_file_name)

    X_train = pd.read_csv(X_train_file_path)
    y_train = pd.read_csv(y_train_file_path)

    #calling the function
    #lstm_model = training_models(X_train,y_train,'LSTM')
    xgb_model = training_models(X_train,y_train,'XGBoost')
