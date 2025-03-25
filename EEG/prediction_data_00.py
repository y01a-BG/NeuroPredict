import os
import pandas as pd
import numpy as np

#################################################################################
################## VALID FOR ALL MODELS ########################################
################## Run it ones and then use data for all models to be able to compare
########################################################################################

def prediction_data(data: pd.DataFrame, exclude_colums: list = [4,5]) -> pd.DataFrame:
    """
    Clean raw data by
    - dropping unneeded columns
    - taking only columns we need
    - taking out X_pred,y_pred and leaves data without these 2x3 classes 0r 2x2 classes data
    """

   # (1) getting rid of not-needed column
    if 'Unnamed' in data.columns:
        data.drop(columns=['Unnamed'], inplace=True)


    #(2) Taking only classes (1,2,3)
    if exclude_colums ==[4,5]:
        data = data[~data['y'].isin([4, 5])]
        print(f"✅ You will work with 3 classes: tumor seizure, tumor base, healthy base")
    elif exclude_colums ==[1,4,5]:
        data = data[~data['y'].isin([1,4, 5])]
        print(f"✅ You will work with 2 classes: tumor base, healthy base")


    # Assuming there's a 'y' column that differentiates the classes
    prediction_data = pd.concat([
        data[data['y'] == c].iloc[[0, -1]] for c in data['y'].unique()])

    # Remove the selected rows from the original dataset
    data = data.drop(prediction_data.index)

    # Save the modified dataset and prediction data
    output_path = "./processed_data"
    data_file_name = "data.csv"
    data__output_file = os.path.join(output_path, data_file_name)
    data.to_csv(data__output_file, index=False)

    # Save prediction  data
    prediction_data_file_name = "prediction_data.csv"
    prediction_data_output_file = os.path.join(output_path, prediction_data_file_name)
    prediction_data.to_csv(prediction_data_output_file, index=False)


    print(f"✅ Successfully extracted 2 samples per class and saved data with {prediction_data.shape} columns less!")
    print(f"✅ New data size {data.shape}")
    return data, prediction_data

###########################################
##########################################

if __name__ == "__main__":

    input_path = "./raw_data"
    file_name = "Epileptic Seizure Recognition.csv"
    file_path = os.path.join(input_path, file_name)
    data = pd.read_csv(file_path)

    data, prediction_data = prediction_data(data,exclude_colums = [4,5])
