import os
import pandas as pd
import numpy as np

from encoding_03 import encoder_LSTM
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report

#######################################################################
####### Load test  data from processed_data   ####################
######################################################################

input_path = "./NeuroPredict/processed_data"
X_test_file_name = "X_test.csv"
y_test_file_name = "y_test.csv"
X_test_file_path = os.path.join(input_path, X_test_file_name)
y_test_file_path = os.path.join(input_path, y_test_file_name)


X_test = pd.read_csv(X_test_file_path)
y_test = pd.read_csv(y_test_file_path)

X_test, y_test = encoder_LSTM(X_test,y_test)


###################################################
######### Load model & test data  ################
#################################################

model = load_model("models/LSTMmodel.h5")


# Assuming model.predict(X_test) gives probabilities for each class
y_pred_probs = model.predict(X_test)

# Convert probabilities to class labels (index of the max probability)
y_pred_labels = np.argmax(y_pred_probs, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Calculate accuracy score
accuracy = accuracy_score(y_test_labels, y_pred_labels)
precision = precision_score(y_test_labels, y_pred_labels, average='macro')  # 'macro' averages metrics across all classes
recall = recall_score(y_test_labels, y_pred_labels, average='macro')
f1 = f1_score(y_test_labels, y_pred_labels, average='macro')

# Confusion matrix
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
# Create classification report (including precision, recall, f1-score per class)
class_report = classification_report(y_test_labels, y_pred_labels, target_names=['Class 0', 'Class 1', 'Class 2'])

# Display results as a table
results = {
    'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'],
    'Score': [accuracy, precision, recall, f1]
}

# Create DataFrame for easy display
results_df = pd.DataFrame(results)

print("✅ Metrics Table:")
print(results_df)

print("✅ \nConfusion Matrix:")
print(conf_matrix)

print("✅ \nClassification Report:")
print(class_report)

#print(f"✅ Accuracy: {accuracy}")
