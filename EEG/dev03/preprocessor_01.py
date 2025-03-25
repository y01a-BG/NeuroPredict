import os
import pandas as pd
import numpy as np

##################################################
################## VALID FOR ALL MODELS ###########
##################################################

def preprocess_data(data: pd.DataFrame, data_path) -> pd.DataFrame:
    """
    Clean raw data by
    - dropping unneeded columns
    - taking only columns we need
    """

   # (1) getting rid of not-needed column
    if 'Unnamed' in data.columns:
        data.drop(columns=['Unnamed'], inplace=True)

    #(2) Taking only cats (1,2,3)
    data = data[~data['y'].isin([4, 5])]

    data.to_csv(data_path, index=False)


    print(f"✅ Data preprocessed and saved to: {data_path}")
    return data


###########################################
##########################################

if __name__ == "__main__":

    input_path = "./raw_data"
    file_name = "Epileptic Seizure Recognition.csv"
    file_path = os.path.join(input_path, file_name)

    output_path = "./processed_data"
    data_file = "data.csv"
    data_path = os.path.join(output_path, data_file)
    data = pd.read_csv(file_path)

    data = preprocess_data(data, data_path)
