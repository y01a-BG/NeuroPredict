import os
import pandas as pd
import numpy as np
import xgboost as xgb


from preprocessor_01 import preprocess_data
from test_and_train_split_02 import test_train_split_save
from encoding_03 import encoder_XGBoost
from modeling_04 import initialize_xgb_model
from training_05 import train_xgb_model
from evaluating_06 import evaluate_xgbmodel
#from predictor_07 import pred_xgb

input_path = "./raw_data"
file_name = "Epileptic Seizure Recognition.csv"
file_path = os.path.join(input_path, file_name)

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
X_train, X_test, y_train, y_test, random_test_samples = test_train_split_save(X,y,output_path)

#### encoding_03 for XGBoost
X_train,y_train = encoder_XGBoost(X_train,y_train)
X_train = encoder_XGBoost(X_train)

### modeling_04 for XGBoost
input_shape = (178,1)
model_xgb = initialize_xgb_model()

##### training_05
best_xgb = train_xgb_model(X_train,y_train)

# Save XBGoost  trained model
best_xgb.save_model("models/XGBoost_model.json")

# evaluating_06
results_df = evaluate_xgbmodel(X_test, y_test)

#predicting_07
#pred_dictionary = pred_xgb(X_pred)
