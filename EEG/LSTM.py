import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import LSTM, BatchNormalization, Dropout, Dense, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import callbacks

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report


#######################################
######## preprocess_01 ###############
#####################################

output_path = "./processed_data"
data_file = "data.csv"
data_path = os.path.join(output_path, data_file)
data = pd.read_csv(data_path)

X = data.drop(columns = 'y')
y = data.y
print(f"✅ Data separated (X,y)")

#######################################
######## X,y_test_train_02 ###############
#####################################



# Split the data into 80% train and 20% test, with random_state=42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_file_name = "X_train.csv"
X_train_output_file = os.path.join(output_path, X_train_file_name)
X_train.to_csv(X_train_output_file, index=False)

y_train_file_name = "y_train.csv"
y_train_output_file = os.path.join(output_path, y_train_file_name)
y_train.to_csv(y_train_output_file, index=False)

X_test_file_name = "X_test.csv"
X_test_output_file = os.path.join(output_path, X_test_file_name)
X_test.to_csv(X_test_output_file, index=False)

y_test_file_name = "y_test.csv"
y_test_output_file = os.path.join(output_path, y_test_file_name)
y_test.to_csv(y_test_output_file, index=False)

print("✅ Data split into train/test and saved as .csv")


#######################################
######## encoding_03  ###############
#####################################

X_train_enc = np.expand_dims(X_train, axis=2)  # to be able to enter the LSTM model
y_train = y_train - 1  # Shift labels to start from 0
y_train_enc = to_categorical(y_train, num_classes=3)
print(f"✅ Train data encoded for LSTM, with X_train shape : {X_train_enc.shape} and y_train shape; {y_train_enc.shape}")

#####################################################
################### MODEL LSTM  modeling_04 ##########
#####################################################

#Bidirectional LSTM model
input_shape = (178,1)
model = Sequential()

# First Bidirectional LSTM layer with L2 regularization
model.add(Bidirectional(LSTM(128, activation='tanh',
                            return_sequences=True,
                            kernel_regularizer='l2'),
                            input_shape=input_shape)
          )

model.add(BatchNormalization())
model.add(Dropout(0.3))

# Second Bidirectional LSTM layer with L2 regularization
model.add(Bidirectional(LSTM(128, activation='tanh',
                             return_sequences=True,
                            kernel_regularizer='l2')))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Third LSTM layer with fewer units and L2 regularization
model.add(LSTM(64, activation='tanh', kernel_regularizer='l2'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Dense layer with ReLU activation
model.add(Dense(64, activation='relu'))

# Output layer with softmax activation for multi-class classification
model.add(Dense(3, activation='softmax'))

# Compile the model with the Adam optimizer and sparse categorical cross-entropy loss
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print("✅ Improved LSTM Model1 initialized")

#####################################################
################### Fitting LSTM  training_05 ########
#####################################################


es = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

history = model.fit(
        X_train_enc,y_train_enc,
        validation_split=0.2,
        epochs=50,
        batch_size=64,
        callbacks=[es],
        verbose=1
    )

print(f"✅ LSTM Model trained on {len(X_train_enc)} rows with min val MAE: {round(np.min(history.history['val_loss']), 2)}")


#####################################################
################### Evaluating LSTM  evaluating_06 ########
#####################################################


X_test_enc = np.expand_dims(X_test, axis=2)  # to be able to enter the LSTM model
y_test = y_test - 1  # Shift labels to start from 0
y_test_enc = to_categorical(y_test, num_classes=3)
print(f"✅ Test data encoded for LSTM, with X_test shape : {X_test_enc.shape} and y_test shape; {y_test_enc.shape}")

# Assuming model.predict(X_test) gives probabilities for each class
y_pred_probs = model.predict(X_test_enc)

# Convert probabilities to class labels (index of the max probability)
y_pred_labels = np.argmax(y_pred_probs, axis=1)
y_test_labels = np.argmax(y_test_enc, axis=1)

# Calculate accuracy score
accuracy = accuracy_score(y_test_labels, y_pred_labels)
precision = precision_score(y_test_labels, y_pred_labels, average='macro')  # 'macro' averages metrics across all classes
recall = recall_score(y_test_labels, y_pred_labels, average='macro')
f1 = f1_score(y_test_labels, y_pred_labels, average='macro')

# Display results as a table
results = {
        'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'],
        'Score': [accuracy, precision, recall, f1]
}
# Create DataFrame for easy display
results_df = pd.DataFrame(results)

# Confusion matrix
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
print("✅ \nConfusion Matrix:")
print(conf_matrix)
# Create classification report (including precision, recall, f1-score per class)
class_report = classification_report(y_test_labels, y_pred_labels, target_names=['Class 0', 'Class 1', 'Class 2'])
print("✅ \nClassification Report:")
print(class_report)

print(f"✅ Metrics Table:{results_df}")

############################################
 ########  Save LSTM  trained model  ########
 ############################################


model.save("models/model_lstm.h5")
print(f"✅  LSTM model saved")


#####################################################
################### Evaluating LSTM  predictor_07 ########
#####################################################

input_path = "./processed_data"
X_pred_file_name = "random_test_samples.csv"
X_pred_file_path = os.path.join(input_path, X_pred_file_name)
X_pred = pd.read_csv(X_pred_file_path)

X_pred_enc = np.expand_dims(X_pred, axis = 2)  # to be able to enter the LSTM model
y_pred_probs = model.predict(X_pred_enc)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

predictions = []
for label in y_pred_labels:
    if label == 0:
        prediction_text = "Your EEG is predicted to be tumor-induced seizure EEG."
        predictions.append(prediction_text)
    elif label == 1:
        prediction_text = "Your EEG is predicted to be tumor baseline EEG."
        predictions.append(prediction_text)
    else:
        prediction_text = "Your EEG is predicted to be healthy baseline EEG."
        predictions.append(prediction_text)


pred_dictionary = {"predictions": predictions}
print(f'✅ Prediction results for 6 EEG samples are : {pred_dictionary}')






def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('accuracy')
    ax[1].plot(history.epoch, history.history["accuracy"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()
    plt.show()

plot_history(history)
