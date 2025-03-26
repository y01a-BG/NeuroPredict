import os
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier



##################################################
########### BinaryXGBoost Classifier #############
##################################################

#######################################
######## preprocess_01 ###############
#####################################

output_path = "./processed_data"
data_file = "data.csv"
data_path = os.path.join(output_path, data_file)
data = pd.read_csv(data_path)

#Exclude seasure, leave only 2,3
data = data[~data['y'].isin([1])]
X = data.drop(columns = 'y')
y = data.y
y = y-2
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

###############################################################
##############   encoding_03  ################################
##############################################################

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

X_train_enc = from_2d_array_to_nested(X_train,
                                                    index=None, columns=None, time_index=None,
                                                    cells_as_numpy=False
                                )
X_test_enc = from_2d_array_to_nested(X_test,
                                index=None, columns=None, time_index=None,
                                cells_as_numpy=False
                    )
print("✅ X_train / X_test is encoded for MiniRocketTransform.")

####################################################
#################  MODEL    Rocket-Transform + XGBoost ##################
######################################################


rocket = MiniRocket(n_jobs=-1)
X_train_transformed = rocket.fit_transform(X_train_enc)
X_test_transformed = rocket.transform(X_test_enc)

clf = XGBClassifier(n_estimators=200, learning_rate=0.05)
clf.fit(X_train_transformed, y_train)
y_pred = clf.predict(X_test_transformed)


#########################################################################
#################  MODEL    EVALUATION       ##################
#########################################################################


# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # 'macro' averages metrics across all classes
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Display results as a table
results = {
        'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'],
        'Score': [accuracy, precision, recall, f1]
    }
# Create DataFrame for easy display
results_df = pd.DataFrame(results)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("✅ \nConfusion Matrix:")
print(conf_matrix)
# Create classification report (including precision, recall, f1-score per class)
class_report = classification_report(y_test, y_pred, target_names=['Class 1', 'Class 2'])
print("✅ \nClassification Report:")
print(class_report)
print(f"✅ Metrics Table:{results_df}")
print(f"✅ Metrics Table:{results_df}")



#######################################################################
#################### Saving BinaryXGBoost model ############################
######################################################################

# Save BinaryXGBoost  trained model
clf.save_model("./models/binary_xgb.json")


#######################################################################
#################### Predicting with  RocketClassifier ###############
######################################################################

output_path = "./processed_data"
pred_file = "prediction_data.csv"
pred_path = os.path.join(output_path, pred_file)
data_pred = pd.read_csv(pred_path)

#Exclude seasure, leave only 2,3
data_pred = data_pred[~data_pred['y'].isin([1])]
X_pred = data_pred.drop(columns = 'y')
y_pred_original = data_pred.y
print(f"✅ Prediction data separated (X,y)")

X_pred_enc = from_2d_array_to_nested(X_pred,
                                index=None, columns=None, time_index=None,
                                cells_as_numpy=False
                    )
X_pred_transformed = rocket.transform(X_pred_enc)

y_pred = clf.predict(X_pred_transformed)

predictions = []
for label in y_pred:
    if label == 0:
        prediction_text = "Your EEG is predicted to be tumor baseline EEG."
        predictions.append(prediction_text)
    else:
        prediction_text = "Your EEG is predicted to be healthy baseline EEG."
        predictions.append(prediction_text)


pred_dictionary = {"predictions": predictions}
print(f'✅ Prediction results for 4 EEG samples are : {pred_dictionary}')




##########################################################################
############# Optional PLotting ##########################################
#########################################################################
#Learning curve

# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.model_selection import learning_curve

# # Compute learning curve
# train_sizes, train_scores, test_scores = learning_curve(
#     rocket, X_test_enc, y_train, cv=5, scoring="accuracy", n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
# )

# # Compute mean and std
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)

# # Plot learning curve
# plt.figure(figsize=(8, 6))
# plt.plot(train_sizes, train_mean, 'o-', label="Train Accuracy", color="blue")
# plt.plot(train_sizes, test_mean, 'o-', label="Test Accuracy", color="red")

# plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.1)
# plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="red", alpha=0.1)

# plt.xlabel("Training Set Size")
# plt.ylabel("Accuracy")
# plt.title("Learning Curve: Train vs Test Accuracy")
# plt.legend()
# plt.grid()
# plt.show()
