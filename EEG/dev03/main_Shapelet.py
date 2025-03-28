import os
import pandas as pd
import numpy as np
import sktime
import joblib


from EEG.dev03.preprocessor_01 import preprocess_data
from EEG.dev03.X_y_test_train_02 import test_train_split_save
from EEG.dev03.encoding_03 import from_2d_array_to_nested
from EEG.dev03.modeling_04 import initialize_shapelet_model
from EEG.dev03.training_05 import train_shapelet_model
from EEG.dev03.evaluating_06 import evaluate_shapelet_model
#from predictor_07 import pred_lstm


input_path = "./raw_data"
file_name = "Epileptic Seizure Recognition.csv"
file_path = os.path.join(input_path, file_name)

output_path = "./processed_data"
data_file = "data.csv"
data_path = os.path.join(output_path, data_file)
data = pd.read_csv(file_path)

X_pred_file_name = "random_test_samples.csv"
X_pred_file_path = os.path.join(output_path, X_pred_file_name)

X_pred = pd.read_csv(X_pred_file_path)

#######preprocessor_01
data = preprocess_data(data, data_path)
X = data.drop(columns = 'y')
y = data.y

print(f"✅ Data preprocessed and split to (X,y)")


####### test_and_train_split_02
X_train, X_test, y_train, y_test = test_train_split_save(X,y,output_path)

#### encoding_03 for Rocket and Shapelet
X_train_enc =from_2d_array_to_nested(X_train)
X_test_enc = from_2d_array_to_nested(X_test)


### modeling_04 for Shapelet
shapelet = initialize_shapelet_model()

##### training_05
shapelet_model = train_shapelet_model(X_train_enc, y_train, shapelet)

# Save Shapelet  trained model
joblib.dump(shapelet_model, "./models/SHAPELETmodel.pkl")

# evaluating_06
results_df = evaluate_shapelet_model(X_test_enc, y_test)
