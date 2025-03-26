import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


import tensorflow as tf
from keras.utils import to_categorical
from keras.models import load_model


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report

from sktime.datatypes._panel._convert import from_2d_array_to_nested
from sktime.transformations.panel.rocket import MiniRocket

from xgboost import XGBClassifier


############################################
####### Loading models of interest ########
###########################################

# Load trianed models
lstm = load_model("./models/model_lstm.h5")
binary_xgb = XGBClassifier()
binary_xgb.load_model("./models/binary_xgb.json")

####################################################
####### Loading/ LSTM - encoding test data   ########
####################################################

X_test = pd.read_csv("./processed_data/X_test.csv")
y_test = pd.read_csv("./processed_data/y_test.csv")
print(f"✅ X_test, y_test loaded")

X_test_lstm = np.expand_dims(X_test, axis=2)  # to be able to enter the LSTM model
y_test = y_test -1
y_test_lstm = to_categorical(y_test, num_classes=3)

#print(y_test)
print(f"✅ X_test_lstm, y_test_lstm  are ready.")

####################################################
####### Evaluating the hybrid model  ########
####################################################

def hybrid_evaluation(X_test_lstm, lstm, binary_xgb):
    """
    X_test_lstm -> input data for LSTM
    X_test_rocket -> input data for ROCKET (only Class 1 & 2)
    lstm -> Trained LSTM model
    binary_rocket -> Trained ROCKET model
    """
    # Step 1: Get LSTM predictions
    y_lstm_probs = lstm.predict(X_test_lstm)  # Assume this gives probabilities
    y_lstm_preds = np.argmax(y_lstm_probs, axis=1)  # Predicted classes from LSTM

    #breakpoint()
    # Debug
    print(f"Unique values in y_lstm_preds before conversion: {np.unique(y_lstm_preds)}")
    print(f'Classes population after lstm : {pd.DataFrame(y_lstm_preds).value_counts()}')

    # Define the category to exclude
    category_to_exclude = 0  # Replace with actual category
    # Create a mask for the rows to keep
    mask = y_lstm_preds != category_to_exclude
    # With maskcreate X_test_xgb, y_test_xgb
    X_test_xgb = X_test[mask]
    #y_test_xgb = y_test[mask]

    # encoding data for XGBoost
    X_test_enc = from_2d_array_to_nested( X_test_xgb,
                                                    index=None, columns=None, time_index=None,
                                                    cells_as_numpy=False
                                )


    print("✅ X_train / X_test is encoded for MiniRocketTransform.")
    ###################
    #  MiniRocket transform
    ################
    rocket = MiniRocket(n_jobs=-1)
    X_test_transformed = rocket.fit_transform(X_test_enc)

    xgb_preds = binary_xgb.predict(X_test_transformed)

    # Step 4: Replace LSTM predictions with ROCKET predictions for Class 1 & 2 cases
    y_lstm_preds[mask] = xgb_preds +1

    # Step 5: Ensure all predictions are within the valid range of classes [0, 1, 2]
    y_lstm_preds = np.clip(y_lstm_preds, 0, 2)  # Ensure the class predictions are within [0, 2]

    return y_lstm_preds


y_hybr_preds = hybrid_evaluation(X_test_lstm,lstm, binary_xgb)

########################################################################
# Evaluation Metrix:

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

print(f"✅ Prediction Data separated (X_pred,y_pred)")


X_pred_enc = np.expand_dims(X_pred, axis=2)


def hybrid_prediction(X_pred_enc, lstm, binary_xgb):
    """
    X_test_lstm -> input data for LSTM
    X_test_rocket -> input data for ROCKET (only Class 1 & 2)
    lstm -> Trained LSTM model
    binary_rocket -> Trained ROCKET model
    """
    # Step 1: Get LSTM predictions
    y_lstm_probs = lstm.predict(X_pred_enc)  # Assume this gives probabilities
    y_lstm_preds = np.argmax(y_lstm_probs, axis=1)  # Predicted classes from LSTM

    # Debug
    print(f"Unique values in y_lstm_preds before conversion: {np.unique( y_lstm_preds)}")
    print(f'Classes population after lstm : {pd.DataFrame(y_lstm_preds).value_counts()}')

    # Define the category to exclude
    category_to_exclude = 0  # Replace with actual category
    # Create a mask for the rows to keep
    mask = y_lstm_preds != category_to_exclude
    # With mask create X_test_xgb, y_test_xgb
    X_pred_xgb = X_pred[mask]
    y_pred_xgb = y_pred[mask]

    # encoding data for XGBoost
    X_pred_xgb_enc = from_2d_array_to_nested(X_pred_xgb,
                                                    index=None, columns=None, time_index=None,
                                                    cells_as_numpy=False
                                )


    print("✅ X_pred_xgb is encoded for MiniRocketTransform.")
    ###################
    #  MiniRocket transform
    ################
    rocket = MiniRocket(n_jobs=-1)
    X_pred_xgb_transformed = rocket.fit_transform(X_pred_xgb_enc)

    xgb_preds = binary_xgb.predict(X_pred_xgb_transformed)

    # Step 4: Replace LSTM predictions with XGBoost  predictions for Class 1 & 2 cases
    y_lstm_preds[mask] = xgb_preds + 1

    # Step 5: Ensure all predictions are within the valid range of classes [0, 1, 2]
    y_lstm_preds = np.clip(y_lstm_preds, 0, 2)  # Ensure the class predictions are within [0, 2]

    return y_lstm_preds


y_pred_preds = hybrid_prediction(X_pred_enc,lstm, binary_xgb)

print("Final predictions:",y_pred_preds)
