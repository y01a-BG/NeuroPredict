from preprocessor_01 import load_and_preprocess_data
from encoding_03 import create_xgb_model
from modeling_04 import perform_grid_search

def train_model(file_path):
    """
    Loads data, trains the model using grid search,
    and returns the best estimator, test data, and scaler.
    """
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(file_path)
    xgb = create_xgb_model()
    grid_search = perform_grid_search(xgb, X_train, y_train, X_test, y_test)

    print("Best Parameters from Grid Search:", grid_search.best_params_)
    optimized_xgb = grid_search.best_estimator_

    return optimized_xgb, X_test, y_test, scaler
