# packagename/Rocket/ShapeletClassifier.py

# importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sktime.classification.shapelet_based import ShapeletTransformClassifier

# Load data and preprocess
data = pd.read_csv('../raw_data/Epileptic Seizure Recognition.csv')
data.drop(columns=['Unnamed'], inplace=True)
df = data[~data['y'].isin([4, 5])]
X = df.drop(columns='y')
y = df.y

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Helper function to convert 2D dataframe to nested format for sktime classifiers
def from_2d_array_to_nested(X, index=None, columns=None, time_index=None, cells_as_numpy=False):
    if (time_index is not None) and cells_as_numpy:
        raise ValueError(
            "`time_index` cannot be specified when `cells_as_numpy` is True, "
            "time index can only be set to pandas Series"
        )
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    container = np.array if cells_as_numpy else pd.Series
    n_instances, n_timepoints = X.shape

    if time_index is None:
        time_index = np.arange(n_timepoints)
    kwargs = {"index": time_index}

    Xt = pd.DataFrame(
        pd.Series([container(X[i, :], **kwargs) for i in range(n_instances)])
    )
    if index is not None:
        Xt.index = index
    if columns is not None:
        Xt.columns = columns
    return Xt

# Convert data into nested format for Shapelet classifier
X_train_nested = from_2d_array_to_nested(X_train)
X_test_nested = from_2d_array_to_nested(X_test)

# Initialize and fit Shapelet classifier
shapelet_class = ShapeletTransformClassifier(n_shapelet_samples=1000,
                                             max_shapelets=None, max_shapelet_length=None,
                                             estimator=None, transform_limit_in_minutes=0,
                                             time_limit_in_minutes=0,
                                             save_transformed_data=False,
                                             n_jobs=1, batch_size=100, random_state=None)

# Train the classifier
shapelet_class.fit(X_train_nested, y_train)

# Predict with the trained model
y_pred_shapelet = shapelet_class.predict(X_test_nested)

# Evaluate the Shapelet classifier
conf_matrix_shapelet = confusion_matrix(y_test, y_pred_shapelet)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_shapelet, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Shapelet Classifier')
plt.show()

# Display classification report
class_report_shapelet = classification_report(y_test, y_pred_shapelet)
print("Classification Report - Shapelet Classifier:\n", class_report_shapelet)

# Save the trained model using joblib
joblib.dump(shapelet_class, 'shapelet_classifier_model.joblib')
print("Model successfully saved as shapelet_classifier_model.joblib")
