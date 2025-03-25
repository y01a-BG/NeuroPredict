import os
import pandas as pd
import numpy as np

#################################################################################
################## VALID FOR ALL MODELS ########################################
################## Run it ones and then use data for all models to be able to compare
########################################################################################

def random_samples_data(data: pd.DataFrame,exclude_colums: list = [4,5]) -> pd.DataFrame:
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
        data = data[~data['y'].isin([4, 5])]
        print(f"✅ You will work with 2 classes: tumor base, healthy base")



    # Assuming the class label column is named 'y'
    selected_samples = data.groupby("y", group_keys=False).apply(lambda x: x.sample(2, random_state=42))

    # Remove the selected samples from the original dataset
    data_remaining = data.drop(selected_samples.index)

    # Save the modified dataset and prediction data
    output_path = "./processed_data"
    data_file_name = "data.csv"
    data__output_file = os.path.join(output_path, data_file_name)
    data_remaining.to_csv(data__output_file, index=False)

    # Save prediction  data
    random_data_file_name = "random_test_samples.csv"
    random_data_output_file = os.path.join(output_path, random_data_file_name)
    selected_samples.to_csv(random_data_output_file, index=False)

    print(f"✅ Successfully extracted 2 samples per class and saved data with {selected_samples.shape} columns less!")
    print(f"✅ New data size {data_remaining.shape}")
    return data_remaining, selected_samples

###########################################
##########################################

if __name__ == "__main__":

    input_path = "./raw_data"
    file_name = "Epileptic Seizure Recognition.csv"
    file_path = os.path.join(input_path, file_name)
    data = pd.read_csv(file_path)


    data_remaining, selected_samples = random_samples_data(data, exclude_colums = [4,5])
