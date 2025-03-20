from training_05 import train_model
from evaluating_06 import evaluate_model
from predictor_07 import predict_eeg_class

if __name__ == "__main__":
    # File path to the dataset
    file_path = "Epileptic Seizure Recognition (1).csv"

    # Train the model via grid search
    optimized_xgb, X_test, y_test, scaler = train_model(file_path)

    # Evaluate the optimized model
    evaluate_model(optimized_xgb, X_test, y_test)

    # Example usage: Predict on the first test sample
    sample_eeg_data = X_test[0]
    prediction = predict_eeg_class(sample_eeg_data, scaler, optimized_xgb)
    print("Predicted Class:", prediction)
