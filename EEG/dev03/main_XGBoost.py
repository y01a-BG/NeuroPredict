import os
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split

from EEG.dev03.preprocessor_01 import preprocess_data
from EEG.dev03.X_y_test_train_02 import test_train_split_save
from EEG.dev03.encoding_03 import encoder_XGBoost
from EEG.dev03.modeling_04 import initialize_xgb_model
from EEG.dev03.training_05 import train_xgb_model
from EEG.dev03.evaluating_06 import evaluate_xgbmodel
#from predictor_07 import pred_xgb

from xgboost import XGBClassifier



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
X_train, X_test, y_train, y_test  = test_train_split_save(X,y,output_path)

#### encoding_03 for XGBoost
X_train,y_train = encoder_XGBoost(X_train,y_train)
X_test, y_test = encoder_XGBoost(X_test, y_test)

### modeling_04 for XGBoost
xgb = initialize_xgb_model()

# Validation data need to be given explicitely: split the data into training and validation sets
X_train1, X_val, y_train1, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

##### training_05
XGBmodel = train_xgb_model(X_train1, X_val, y_train1, y_val, xgb)

# Save XBGoost  trained model
XGBmodel.save_model("./models/XGBmodel.json")

# evaluating_06
results_df = evaluate_xgbmodel(X_test, y_test)

#predicting_07
#pred_dictionary = pred_xgb(X_pred)
