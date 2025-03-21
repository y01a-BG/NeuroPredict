import os
import pandas as pd
import numpy as np

from encoding_03 import encoder_LSTM, encoder_XGBoost
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report


#****************************************************************#
#****************************************************************#
#****************************************************************#
#   THIS .py CONTAINS evaluations FOR  DIFFERENT ML/DL -models:
#   1. LSTM
#   2. XG Boost
#   3.
#****************************************************************#
#****************************************************************#
#****************************************************************#





#######################################################################
#################### LSTM model ######################################
######################################################################

def evaluate_lstm(X_test: pd.DataFrame, y_test: pd.DataFrame):
    # Encoding the data for LSTM model
    X_test_enc, y_test_enc = encoder_LSTM(X_test,y_test)
    # Load trianed model from model/ .h5 file
    model = load_model("models/LSTMmodel.h5")
    # Assuming model.predict(X_test) gives probabilities for each class
    y_pred_probs = model.predict(X_test_enc)

    # Convert probabilities to class labels (index of the max probability)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_test_labels = np.argmax(y_test_enc, axis=1)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    precision = precision_score(y_test_labels, y_pred_labels, average='macro')  # 'macro' averages metrics across all classes
    recall = recall_score(y_test_labels, y_pred_labels, average='macro')
    f1 = f1_score(y_test_labels, y_pred_labels, average='macro')

    # Display results as a table
    results = {
        'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'],
        'Score': [accuracy, precision, recall, f1]
    }
    # Create DataFrame for easy display
    results_df = pd.DataFrame(results)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
    print("✅ \nConfusion Matrix:")
    print(conf_matrix)
    # Create classification report (including precision, recall, f1-score per class)
    class_report = classification_report(y_test_labels, y_pred_labels, target_names=['Class 0', 'Class 1', 'Class 2'])
    print("✅ \nClassification Report:")
    print(class_report)


    print(f"✅ Metrics Table:{results_df}")
    return results_df



#######################################################################
#################### XGBoost model ######################################
######################################################################


def evaluate_xgbmodel(X_test, y_test):
    """
    Computes and prints the confusion matrix and classification report.
    """
    # Encoding the data for XGBoost; this is StandardScaler - output data are np.array
    X_test_enc, y_test_enc = encoder_XGBoost(X_test, y_test)

    # Load trained model from model/ .h5 file
    xgb_model = load_model("models/XGBMmodel.h5")

    #Prediction
    y_pred = xgb_model.predict(X_test_enc)




    # Calculate accuracy score
    accuracy = accuracy_score(y_test_enc, y_pred)
    precision = precision_score(y_test_enc, y_pred, average='macro')  # 'macro' averages metrics across all classes
    recall = recall_score(y_test_enc, y_pred, average='macro')
    f1 = f1_score(y_test_enc, y_pred, average='macro')

    # Display results as a table
    results = {
        'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'],
        'Score': [accuracy, precision, recall, f1]
    }
    # Create DataFrame for easy display
    results_df = pd.DataFrame(results)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test_enc, y_pred)
    print("✅ \nConfusion Matrix:")
    print(conf_matrix)
    # Create classification report (including precision, recall, f1-score per class)
    class_report = classification_report(y_test_enc, y_pred, target_names=['Class 0', 'Class 1', 'Class 2'])
    print("✅ \nClassification Report:")
    print(class_report)


    print(f"✅ Metrics Table:{results_df}")
    return results_df


#################################################
######### if runing: python  modelin_04.py ######
#################################################

def evaluate_models(X_test,y_test,model: str):
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
        results_df = evaluate_lstm(X_test, y_test)
        print("✅ LSTM Model evaluated on (X_test, y_test)")
        return results_df


    elif model == "XGBoost":
        evaluate_xgbmodel(X_test, y_test)
        print("✅ XGBoost Model evaluated on (X_test, y_test)")
       # return xgb
    else:
        print(f"Unknown model: {model}. Please choose either 'LSTM' or 'XGBoost'.")



if __name__ == "__main__":

####### Load TEST  data from processed_data   ####################
######## Valid for all models #######################################
    input_path = "./processed_data"
    X_test_file_name = "X_test.csv"
    y_test_file_name = "y_test.csv"
    X_test_file_path = os.path.join(input_path, X_test_file_name)
    y_test_file_path = os.path.join(input_path, y_test_file_name)

    X_test = pd.read_csv(X_test_file_path)
    y_test = pd.read_csv(y_test_file_path)

    #calling a function
   # lstm_eval = evaluate_models(X_test,y_test,'LSTM')
    xgb_eval = evaluate_models(X_test,y_test,'XGBoost')
