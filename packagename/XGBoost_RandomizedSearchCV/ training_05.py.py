from preprocessor_01 import load_and_preprocess_data
from encoding_03 import create_xgb_model
from modeling_04 import perform_random_search

def train_model(file_path):
    """
    Loads data, creates the model, and performs randomized search.
    Returns the best estimator, test data, and scaler.
    """
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(file_path)
    xgb = create_xgb_model()
    random_search = perform_random_search(xgb, X_train, y_train, X_test, y_test)

    print("Best Parameters from Randomised Search:", random_search.best_params_)
    print("Best F1 Macro Score from CV:", random_search.best_score_)

    optimized_xgb = random_search.best_estimator_
    return optimized_xgb, X_test, y_test, scaler
