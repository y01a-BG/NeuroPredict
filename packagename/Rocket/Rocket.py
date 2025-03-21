# importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier

# Load data and preprocess
data = pd.read_csv('/content/drive/MyDrive/Data Science & AI/Projects/Epileptic Seizure Recognition (1).csv')
data.drop(columns=['Unnamed'], inplace=True)
df = data[~data['y'].isin([4, 5])]
X = df.drop(columns='y')
y = df.y

# Split data into 80% train and 20% test, with random_state=42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Helper function converting 2D dataframe to nested dataframe for sktime classifiers
def from_2d_array_to_nested(X, index=None, columns=None, time_index=None, cells_as_numpy=False):
    """Convert 2D dataframe to nested dataframe."""
    if (time_index is not None) and cells_as_numpy:
        raise ValueError(
            "`Time_index` cannot be specified when `return_arrays` is True, "
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

# Convert data into nested format for Rocket and Shapelet classifiers
X_train_nested = from_2d_array_to_nested(X_train)
X_test_nested = from_2d_array_to_nested(X_test)

# Initialize and fit ROCKET classifier
clf = RocketClassifier(num_kernels=1000, rocket_transform='rocket',
                       max_dilations_per_kernel=32, n_features_per_kernel=4,
                       use_multivariate='auto', n_jobs=-1, random_state=None)

clf.fit(X_train_nested, y_train)
y_pred = clf.predict(X_test_nested)

# Evaluation with confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - ROCKET Classifier')
plt.show()

# Display classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report - ROCKET Classifier:\n", class_report)

# Initialize and fit Shapelet classifier
shapelet_class = ShapeletTransformClassifier(n_shapelet_samples=1000,
                                             max_shapelets=None, max_shapelet_length=None,
                                             estimator=None, transform_limit_in_minutes=0,
                                             time_limit_in_minutes=0,
                                             save_transformed_data=False,
                                             n_jobs=1, batch_size=100, random_state=None)

shapelet_class.fit(X_train_nested, y_train)
y_pred_shapelet = shapelet_class.predict(X_test_nested)

# Evaluate Shapelet classifier
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
