import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
#from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report

#######################################
######## preprocess_01 ###############
#####################################

output_path = "./processed_data"
data_file = "data.csv"
data_path = os.path.join(output_path, data_file)
data = pd.read_csv(data_path)

X = data.drop(columns = 'y')
y = data.y
print(f"✅ Data separated (X,y)")

#######################################
######## X,y_test_train_02 ###############
#####################################



# Split the data into 80% train and 20% test, with random_state=42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_file_name = "X_train.csv"
X_train_output_file = os.path.join(output_path, X_train_file_name)
X_train.to_csv(X_train_output_file, index=False)

y_train_file_name = "y_train.csv"
y_train_output_file = os.path.join(output_path, y_train_file_name)
y_train.to_csv(y_train_output_file, index=False)

X_test_file_name = "X_test.csv"
X_test_output_file = os.path.join(output_path, X_test_file_name)
X_test.to_csv(X_test_output_file, index=False)

y_test_file_name = "y_test.csv"
y_test_output_file = os.path.join(output_path, y_test_file_name)
y_test.to_csv(y_test_output_file, index=False)

print("✅ Data split into train/test and saved as .csv")

#######################################
######## encoding_03  ###############
#####################################

scaler = StandardScaler()
scaler_fit = scaler.fit(X_train)


####################################################
#################  MODEL    XGBoost ##################
######################################################


best_params = {
        'subsample': 0.8,
        'reg_lambda': 1,
        'reg_alpha': 1.0,
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.05,
        'gamma': 0,
        'colsample_bytree': 0.8
        }


xgb = XGBClassifier(
            subsample=best_params['subsample'],
            reg_lambda=best_params['reg_lambda'],
            reg_alpha=best_params['reg_alpha'],
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            gamma=best_params['gamma'],
            colsample_bytree=best_params['colsample_bytree'],

            objective='multi:softprob',
            eval_metric="mlogloss",
            num_class=3,
            verbosity=1,
            random_state=42,
            use_label_encoder=False
        )

print("✅ XGBoost Model initialized")

#####################################################
################### Fitting LSTM  training_05 ########
#####################################################

X_train_enc = scaler_fit.transform(X_train)
y_train_enc = y_train.values-1

print(f"✅ Train data encoded for XGBoost, with X_train shape : {X_train_enc.shape} and y_train shape; {y_train_enc.shape}")

## Training the model (Early Stopping and best model is included in .fit() in XGBoost)
# Validation data need to be given explicitely: split the data into training and validation sets
X_train1, X_val, y_train1, y_val = train_test_split(X_train_enc, y_train_enc, test_size=0.2, random_state=42)


xgb.fit(
        X_train1, y_train1,
        eval_set=[(X_train1, y_train1), (X_val, y_val)],
        verbose=True
    )
print(f"✅ XGBoost Model trained")


#######################################################################
#################### XGBoost evaluation_06   #############################
######################################################################

X_test_enc = scaler_fit.transform(X_test)
y_test_enc = y_test.values-1
print(f"✅ Test data encoded for XGBoost, with X_testshape : {X_test_enc.shape} and y_test shape; {y_test_enc.shape}")

#Prediction
y_pred = xgb.predict(X_test_enc)


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
print(f"✅ Metrics Table:{results_df}")

#######################################################################
#################### Saving XGBoost model ############################
######################################################################

# Save XBGoost  trained model
xgb.save_model("./models/model_xgb.json")

#######################################################################
#################### XGBoost predict_07 ############################
######################################################################

input_path = "./processed_data"
X_pred_file_name = "prediction_data.csv"
X_pred_file_path = os.path.join(input_path, X_pred_file_name)
X_pred = pd.read_csv(X_pred_file_path)

X_pred_enc = scaler_fit.transform(X_pred)
y_pred = xgb.predict(X_pred_enc)

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


pred_dictionary = {"predictions": predictions}
print(f'✅ Prediction results for 6 EEG samples are : {pred_dictionary}')
