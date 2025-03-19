
import pandas as pd
import numpy as np

#input_path = "./NeuroPredict/raw_data"
#output_path = "./NeuroPredict/processed_data"
#file_name = "Epileptic Seizure Recognition.csv"
#file_path = os.path.join(input_path, file_name)


def preprocess_data(file_path, output_path) -> pd.DataFrame:
    """
    Clean raw data by
    - dropping unneeded columns
    - taking only columns we need
    """

    data = pd.read_csv(file_path)

   # (1) getting rid of not-needed column
    if 'Unnamed' in data.columns:
        data.drop(columns=['Unnamed'], inplace=True)

    #(2) Taking only cats (1,2,3)
    data = data[~data['y'].isin([4, 5])]
    data.to_csv(output_path, index=False)


    print("✅ data preprocessed and stored in output_path")
    return data
