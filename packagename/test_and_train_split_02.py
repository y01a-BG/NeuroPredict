
import os
import pandas as pd
import numpy as np

from preprocessor_01 import preprocess_data
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#######################################################################
####### Loading data  from preprocessor.py ####################
####### Splitting into train/test here########################
######################################################################

input_path = "./raw_data"
file_name = "Epileptic Seizure Recognition.csv"
file_path = os.path.join(input_path, file_name)

output_path = "./processed_data"
data_file = "data.csv"
data_path = os.path.join(output_path, data_file)

data = preprocess_data(file_path, data_path)


X = data.drop(columns = 'y')
y = data.y

#X = np.expand_dims(X, axis=2)  # to be able to enter the LSTM model
# classification on y-column
#y = y - 1  # Shift labels to start from 0
#y = to_categorical(y, num_classes=3)


#######################################################################
##################### Data Splitting TEST/TRAIN #######################
##################### Saving to processed_data folder #################
#######################################################################


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

print("✅ data split into train/test and saved as .csv")
