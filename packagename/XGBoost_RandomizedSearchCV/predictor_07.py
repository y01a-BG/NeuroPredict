def predict_eeg_class(eeg_data, scaler, optimized_xgb):
    """
    Predict the class of EEG data using the trained XGBoost model.

    Parameters:
      eeg_data (numpy array): EEG data (shape: (1, n_features)).
      scaler: Preprocessing scaler.
      optimized_xgb: Trained XGBoost model.

    Returns:
      str: Predicted class ('Seizure Activity', 'Tumour Area', 'Healthy Region').
    """
    eeg_data_scaled = scaler.transform(eeg_data.reshape(1, -1))
    predicted_class = optimized_xgb.predict(eeg_data_scaled)[0]
    class_labels = ['Seizure Activity', 'Tumour Area', 'Healthy Region']
    return class_labels[predicted_class]
