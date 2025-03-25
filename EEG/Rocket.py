import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split

from sktime.classification.kernel_based import RocketClassifier
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
print("✅ X_train will be encoded for RocketClassifier.")

####################################################
#################  MODEL    Rocket ##################
######################################################

rocket = RocketClassifier(num_kernels=1000,
                        rocket_transform='rocket',
                        max_dilations_per_kernel=32,
                        n_features_per_kernel=4,
                        use_multivariate='auto',
                        n_jobs=-1,
                        random_state=None
    )


#######################################################################
#################### RocketClassifier training #########################
######################################################################

rocket.fit(X_train_enc, y_train)


#######################################################################
#################### Rocket  evaluating ###############################
######################################################################

X_test_enc = from_2d_array_to_nested(X_test,
                                index=None, columns=None, time_index=None,
                                cells_as_numpy=False
                    )


y_pred = rocket.predict(X_test_enc)

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
class_report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1', 'Class 2'])
print("✅ \nClassification Report:")
print(class_report)
print(f"✅ Metrics Table:{results_df}")
print(f"✅ Metrics Table:{results_df}")



#######################################################################
#################### Saving RocketClassifier ############################
######################################################################

# Save Rocket  trained model
joblib.dump(rocket, "./models/model_rocket.pkl")
