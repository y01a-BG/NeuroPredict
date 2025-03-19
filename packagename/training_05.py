
import os
import pandas as pd
import numpy as np

from modeling_04 import initialize_model, compile_model
from encoding_03 import encoder_LSTM


import tensorflow as tf
from keras import callbacks
from keras.callbacks import EarlyStopping


from colorama import Fore, Style
from typing import Tuple


#######################################################################
####### Load train  data from processed_data   ####################
######################################################################

input_path = "/home/jovana/code/y01a-BG/NeuroPredict/processed_data"
X_train_file_name = "X_train.csv"
y_train_file_name = "y_train.csv"
X_train_file_path = os.path.join(input_path, X_train_file_name)
y_train_file_path = os.path.join(input_path, y_train_file_name)


X_train = pd.read_csv(X_train_file_path)
y_train = pd.read_csv(y_train_file_path)

X_train, y_train = encoder_LSTM(X_train,y_train)
#######################################################################
##################### Calling the compiled model from modeling.py #####
######################################################################

input_shape = (178,1)
model = initialize_model(input_shape)

model = compile_model(model, learning_rate=0.001)


#######################################################################
##################### Define function to TRAIN the model  ###########
######################################################################

def train_model(
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
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=150,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )

    print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_loss']), 2)}")

    return model, history

#######################################################################
##################### Call above function to TRAIN the model  ###########
######################################################################

model,history = train_model(model,
        X_train,y_train,
        batch_size=128,
        patience=1,
        validation_data=None, # overrides validation_split
        validation_split=0.2
        )
# Save trained model
model.save("models/LSTMmodel.h5")
