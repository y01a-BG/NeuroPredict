
import os
import pandas as pd
import numpy as np

from encoding_03 import encoder_LST
import tensorflow as tf
from keras.models import load_model



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

    X_processed = encoder_LST(X_pred)
    y_pred_probs = model.predict(X_processed)
    y_pred_label = np.argmax(y_pred_probs, axis=1)

    if  y_pred_label == 0:
        print("\n✅ prediction done: Your EEG is predicted to be tumor-induced seizure EEG")
    elif  y_pred_label == 0:
        print("\n✅ prediction done: Your EEG is predicted to be tumor baseline EEG")
    else:
        print("\n✅ prediction done: Your EEG is predicted to be healthy baseline EEG")


    return y_pred_label
