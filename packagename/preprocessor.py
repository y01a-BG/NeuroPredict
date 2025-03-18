
import pandas as pd
import numpy as np


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - dropping unneeded columns
    - taking only columns we need
    """

    # Dropping not needed columns
    data.drop(columns=['Unnamed'], inplace=True)

    # Taking only healthy baseline vs. tumor seizure y (1,2,3)
    data = data[~data['y'].isin([4, 5])]


    print("✅ data cleaned")

    return data
