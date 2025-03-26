import os
import pandas as pd
import numpy as np
import joblib

import tensorflow as tf
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report




######################################################
############ Hybrid Predictor #######################
####################################################
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


def hybrid_prediction(X_pred, lstm, binary_rocket):
    """
    X_pred -> New input data
    lstm -> Trained LSTM model
    binary_rocket -> Trained ROCKET model
    Returns final hybrid predictions for X_pred
    """
    X_pred_enc =  np.expand_dims(X_pred, axis = 2)  # to be able to enter the LSTM model
    # Step 1: Get LSTM predictions
    y_lstm_probs = lstm.predict(X_pred_enc)  # Assuming LSTM outputs probability scores
    y_lstm_preds = np.argmax(y_lstm_probs, axis=1)  # Convert to class labels

    # Debugging
    print(f"Unique values in y_lstm_preds before conversion: {np.unique(y_lstm_preds)}")
    print(f'Classes population after lstm : {pd.DataFrame(y_lstm_preds).value_counts()}')
    #breakpoint()
    # Define the category to exclude
    category_to_exclude = 0  # Replace with actual category
    # Create a mask for the rows to keep
    mask = y_lstm_preds != category_to_exclude
    # With maskcreate X_test_rocket, y_test_rocket
    X_pred_rocket = X_pred[mask]
    # encoding data for Rocket
    X_pred_rocket =from_2d_array_to_nested(X_pred_rocket, index=None, columns=None, time_index=None, cells_as_numpy=False)

    # Step 4: Predict with ROCKET for Class 2  and 3 (it gives original lables)
    rocket_preds = binary_rocket.predict(X_pred_rocket)

    # Step 5: Replace LSTM predictions with ROCKET predictions for Class 1 & 2
    y_lstm_preds[mask] = rocket_preds -1

    # Step 6: Ensure valid class predictions [0, 1, 2]
    y_lstm_preds = np.clip(y_lstm_preds, 0, 2)

    return y_lstm_preds


    #################################################
    ####### Evaluating the hybrid model  ########
    ####################################################

    # def hybrid_evaluation(X_test_lstm, lstm, binary_rocket):
    #     """
    #     X_test_lstm -> input data for LSTM
    #     X_test_rocket -> input data for ROCKET (only Class 1 & 2)
    #     lstm -> Trained LSTM model
    #     binary_rocket -> Trained ROCKET model
    #     """
    #     # Step 1: Get LSTM predictions
    #     y_lstm_probs = lstm.predict(X_test_lstm)  # Assume this gives probabilities
    #     y_lstm_preds = np.argmax(y_lstm_probs, axis=1)  # Predicted classes from LSTM

    #     # Debug
    #     print(f"Unique values in y_lstm_preds before conversion: {np.unique(y_lstm_preds)}")
    #     print(f'Classes population after lstm : {pd.DataFrame(y_lstm_preds).value_counts()}')

    #     #breakpoint()

    #     # Define the category to exclude
    #     category_to_exclude = 0 # Replace with actual category
    #     # Create a mask for the rows to keep
    #     mask = y_lstm_preds != category_to_exclude
    #     # With maskcreate X_test_rocket, y_test_rocket
    #     X_test_rocket = X_test[mask]
    #     # y_test_rocket = y_test[mask]
    #     # encoding data for Rocket
    #     X_test_rocket =from_2d_array_to_nested(X_test_rocket, index=None, columns=None, time_index=None, cells_as_numpy=False)

    #     rocket_preds = binary_rocket.predict(X_test_rocket)  # Predict with ROCKET for Class 1 and 2

    #     # Step 4: Replace LSTM predictions with ROCKET predictions for Class 1 & 2 cases
    #     y_lstm_preds[mask] = rocket_preds -1

    #     # Step 5: Ensure all predictions are within the valid range of classes [0, 1, 2]
    #     y_lstm_preds = np.clip(y_lstm_preds, 0, 2)  # Ensure the class predictions are within [0, 2]

    #     return y_lstm_preds


if __name__ == "__main__":

    ############################################
    ####### Loading models of interest ########
    ###########################################

    # Load trianed model from model/ .h5 file
    lstm = load_model("./models/model_lstm.h5")
    binary_rocket = joblib.load("./models/binary_rocket.pkl")

    ####################################################
    ####### Loading/ LSTM - encoding test data   ########
    ####################################################

    X_test = pd.read_csv("./processed_data/X_test.csv")
    y_test = pd.read_csv("./processed_data/y_test.csv")
    y_test = y_test -1 # Shift labels to start from 0
    print(f"✅ X_test, y_test loaded")

    X_test_lstm = np.expand_dims(X_test, axis=2)  # to be able to enter the LSTM model
    y_test_lstm = to_categorical(y_test, num_classes=3)

    #print(y_test)
    print(f"✅ X_test_lstm, y_test_lstm  are ready.")






    y_hybr_preds = hybrid_evaluation(X_test_lstm,lstm, binary_rocket)

    ########################################################################
    # Evaluation Metrix:
    ###########################################################
    # Calculate accuracy score
    accuracy = accuracy_score(y_test,y_hybr_preds)

    # Calculate precision, recall, and f1-score (macro average)
    precision = precision_score(y_test, y_hybr_preds, average='macro')  # 'macro' averages metrics across all classes
    recall = recall_score(y_test, y_hybr_preds, average='macro')
    f1 = f1_score(y_test, y_hybr_preds, average='macro')

    # Display results as a table
    results = {
            'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'],
            'Score': [accuracy, precision, recall, f1]
    }

    # Create DataFrame for easy display
    results_df = pd.DataFrame(results)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_hybr_preds)
    print("✅ \nConfusion Matrix:")
    print(conf_matrix)

    # Create classification report (including precision, recall, f1-score per class)
    class_report = classification_report(y_test, y_hybr_preds, target_names=['Class 0', 'Class 1', 'Class 2'])
    print("✅ \nClassification Report:")
    print(class_report)

    print(f"✅ Metrics Table:\n{results_df}")




    ####################################################
    ############     Prediction Data   ###############
    ####################################################

    data_pred = pd.read_csv("./processed_data/prediction_data.csv")
    X_pred = data_pred.drop(columns = 'y')
    y_pred = data_pred.y


    print(f"✅ Prediction Data separated (X_pred,y_pred), labels {y_pred}")


    X_pred_enc = np.expand_dims(X_pred, axis=2)


    y_pred_hybr = hybrid_prediction(X_pred_enc, lstm,binary_rocket)

    print("Final predictions:", y_pred_hybr)
