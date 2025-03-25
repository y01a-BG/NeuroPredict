
import os
import pandas as pd
import numpy as np

from EEG.dev03.preprocessor_01 import preprocess_data
from sklearn.model_selection import train_test_split



#######################################################################
##################### VALID FOR ALL MODELS   ##########################
##################### Data Splitting TEST/TRAIN #######################
##################### Saving to processed_data folder #################
#######################################################################


def test_train_split_save(X: pd.DataFrame ,y: pd.DataFrame,output_path) -> pd.DataFrame:



    # Split the data into 80% train and 20% test, with random_state=42 for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get 6 random rows from X_test
    #random_test_samples = X_test.copy().sample(n=6, random_state=42)

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


    # Save random test samples
    #random_samples_file_name = "random_test_samples.csv"
    #random_samples_output_file = os.path.join(output_path, random_samples_file_name)
    #random_test_samples.to_csv(random_samples_output_file, index=False)

    print("✅ Data split into train/test and saved as .csv")
    return X_train, X_test, y_train, y_test





###########################################
##########################################

if __name__ == "__main__":

    output_path = "./processed_data"
    data_file = "data.csv"
    data_path = os.path.join(output_path, data_file)
    data = pd.read_csv(data_path)

    data = preprocess_data(data, data_path)
    X = data.drop(columns = 'y')
    y = data.y

    # calling function test_train_split_save(X,y)
    X_train, X_test, y_train, y_test = test_train_split_save(X,y,output_path)

    print("✅ Data split into (X,y) - train/test and  saved as .csv")
