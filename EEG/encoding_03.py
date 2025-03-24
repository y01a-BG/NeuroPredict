import os
import pandas as pd
import numpy as np


from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler




#****************************************************************#
#****************************************************************#
#****************************************************************#
#   THIS .py CONTAINS ENCODING FOR  DIFFERENT ML-models:
#   1. LSTM
#   2. XG Boost
#   3. RocketClassifier and  ShapeletTransformClassifier (from sktime)
#****************************************************************#
#****************************************************************#
#****************************************************************#



############################################################################
########################### LSTM model ####################################
##########################################################################

def encoder_LSTM(X_train: pd.DataFrame, y_train:pd.DataFrame = None) -> pd.DataFrame:

    X_train_enc = np.expand_dims(X_train, axis=2)  # to be able to enter the LSTM model

    if y_train is None:
        print("✅ y_train is not provided, only X_train will be encoded.")
        return X_train_enc
    else:
        y_train = y_train - 1  # Shift labels to start from 0
        y_train_enc = to_categorical(y_train, num_classes=3)

        print(f"✅ Train data encoded for LSTM, with X_train shape : {X_train_enc.shape} and y_train shape; {y_train_enc.shape}")
        return X_train_enc, y_train_enc



############################################################################
########################### XGBoost model ####################################
##########################################################################


def encoder_XGBoost(X_train: pd.DataFrame, y_train: pd.DataFrame = None) -> pd.DataFrame:
    scaler = StandardScaler()
    scaler_fit = scaler.fit(X_train)
    X_train_enc = scaler_fit.transform(X_train)

    if y_train is None:
        print("✅ y_train is not provided, only X_train will be encoded.")
        return X_train_enc

    else:
        y_train_enc = y_train.values-1
        print(f"✅ Train data encoded for XGBoost, with X_train shape : {X_train_enc.shape} and y_train shape; {y_train_enc.shape}")
        return X_train_enc, y_train_enc


###########################################################################
##############    RocketClassifier / ShapeletTransformClassifier    #######
###########################################################################

def from_2d_array_to_nested(X_train: pd.DataFrame, y_train: pd.DataFrame = None, index=None, columns=None, time_index=None, cells_as_numpy=False):
    """Convert 2D dataframe to nested dataframe."""
    if (time_index is not None) and cells_as_numpy:
        raise ValueError(
            "`Time_index` cannot be specified when `return_arrays` is True, "
            "time index can only be set to pandas Series"
        )
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()

    container = np.array if cells_as_numpy else pd.Series
    n_instances, n_timepoints = X_train.shape

    if time_index is None:
        time_index = np.arange(n_timepoints)
    kwargs = {"index": time_index}

    Xt = pd.DataFrame(
        pd.Series([container(X_train[i, :], **kwargs) for i in range(n_instances)])
    )
    if index is not None:
        Xt.index = index
    if columns is not None:
        Xt.columns = columns
    return Xt



#######################################################
######### If runing: python  encoding_03.py ########
######################################################

# model: A string indicating the model type ('LSTM' or 'XGBoost', 'Rocket and Shapelet' - 'RSh')
def encode_data(X_train, y_train, model: str):
    """
    This function will apply encoding based on the model choice (LSTM or XGBoost).

    Parameters:
    - X_train: Input training data
    - y_train: Target training data
    - model: A string indicating the model type  ('LSTM', 'XGBoost', 'Rocket', 'Shapelet')

    Returns:
    - Encoded X_train and y_train based on the selected model.
    """

    if model == "LSTM":
        X_train, y_train = encoder_LSTM(X_train, y_train)
    elif model == "XGBoost":
        X_train, y_train = encoder_XGBoost(X_train, y_train)
    elif model == "RSh":
        X_train = from_2d_array_to_nested(X_train,y_train = None)
    else:
        print(f"Unknown model: {model}. Please choose either: 'LSTM', 'XGBoost', 'Rocket', 'Shapelet'")


    return X_train, y_train


if __name__ == "__main__":

    ####### Load TRAIN data from processed_data   ####################
    ######## Valid for all models #######################################

    input_path = "./processed_data"
    X_train_file_name = "X_train.csv"
    y_train_file_name = "y_train.csv"
    X_train_file_path = os.path.join(input_path, X_train_file_name)
    y_train_file_path = os.path.join(input_path, y_train_file_name)

    X_train = pd.read_csv(X_train_file_path)
    y_train = pd.read_csv(y_train_file_path)

    #calling the function
    #X_train, y_train = encode_data(X_train, y_train, 'LSTM')
    X_train, y_train = encode_data(X_train, y_train, 'XGBoost')
    #X_train = encode_data(X_train,None, 'RSh')
