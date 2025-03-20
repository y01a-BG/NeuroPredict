from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(optimized_xgb, X_test, y_test):
    """
    Evaluates the model on test data and prints the confusion matrix and classification report.
    """
    y_pred = optimized_xgb.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=['Seizure Activity', 'Tumour Area', 'Healthy Region']
    )
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
