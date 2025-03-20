from preprocessor_01 import load_and_preprocess_data

def get_train_test_data(file_path):
    """
    Returns train/test split along with the scaler.
    """
    return load_and_preprocess_data(file_path)
