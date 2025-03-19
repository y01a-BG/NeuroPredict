import os
import pandas as pd
import numpy as np


from keras.utils import to_categorical



def encoder_LSTM(X_train,y_train = None):

    X_train = np.expand_dims(X_train, axis=2)  # to be able to enter the LSTM model

    if y_train is None:
        print("✅ y_train is not provided, only X_train will be encoded.")
        return X_train
    else:
        y_train = y_train - 1  # Shift labels to start from 0
        y_train = to_categorical(y_train, num_classes=3)

        print(f"✅ Train data encoded for LSTM, with X_train shape : {X_train.shape} and y_train shape; {y_train.shape}")
        return X_train, y_train
