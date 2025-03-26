import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.utils import to_categorical

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, BatchNormalization, Dropout, Dense, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import callbacks
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler

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
print("✅ Unique y:", np.unique(y))

# Merging Class 2 and 23 together: tumor/healthy baseline
y = data.y.replace({3: 2})
print("✅ Unique y:", np.unique(y))
y = y -1  # Shift labels to start from 0
print("✅ Unique y:", np.unique(y))
print(f"✅ Data separated X {X.shape} and y {y.shape}")


#######################################
######## X,y_test_train_02 ###############
#####################################

# Split the data into 80% train and 20% test, with random_state=42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the RandomUnderSampler to balance the classes
# Set 'sampling_strategy' to 'auto' (default), which means undersample the majority class to match the minority class size.
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)

# Fit the undersampler on the training data
X_train, y_train = undersampler.fit_resample(X_train, y_train)

# Check the class distribution after undersampling
print("Class distribution in y_train (before undersampling):", pd.Series(y_train).value_counts())
print("Class distribution in y_train_resampled (after undersampling):", pd.Series(y_train).value_counts())

print(f"✅ Data split into train/test {X_train.shape, y_train.shape, X_test.shape, y_test.shape}")


#######################################
######## encoding_03  ###############
#####################################

X_train_enc = np.expand_dims(X_train, axis=2)  # to be able to enter the LSTM model
y_train_enc = np.array(y_train)
print(f"✅ Train data encoded for LSTM, with X_train shape : {X_train_enc.shape} and y_train shape; {y_train.shape}")
print("Unique y_train_enc:", np.unique(y_train_enc))
print("Shape of y_train_enc:", y_train_enc.shape)
#####################################################
################### MODEL LSTM  modeling_04 ##########
#####################################################

# After talking to  ChatGTP about overfitting
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization, Dropout

input_shape = (178, 1)  # EEG input shape

model = Sequential()

# Reduce LSTM layers & units
model.add(LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape))
model.add(BatchNormalization())
model.add(Dropout(0.4))  # Increase dropout

model.add(LSTM(32, activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# Dense layers
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))   # 2 classes

# # Compile with lower learning rate
learning_rate = 0.001  # Reduce LR for stability
optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])

print("✅ Optimized LSTM Model initialized")



#####################################################
################### Fitting LSTM  training_05 ########
#####################################################
# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

# Convert to dictionary format for Keras
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

print(f"✅ Computed class weights: {class_weights_dict}")


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
        batch_size=32,
        callbacks=[es],
        class_weight=class_weights_dict,
        verbose=1
    )

print(f"✅ LSTM Model with weigths trained on {len(X_train_enc)} rows with min val MAE: {round(np.min(history.history['val_loss']), 2)}")


#####################################################
################### Evaluating LSTM  evaluating_06 ########
#####################################################


X_test_enc = np.expand_dims(X_test, axis=2)  # to be able to enter the LSTM model
print(f"✅ Test data encoded for LSTM, with X_test shape : {X_test_enc.shape} and y_test shape; {y_test.shape}")

# Assuming model.predict(X_test) gives probabilities for each class
y_pred_probs = model.predict(X_test_enc)

# Convert probabilities to class labels (index of the max probability)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

# Debug
print(f"Unique values in y_lstm_preds before conversion: {np.unique( y_pred_labels)}")
print(f'Classes population after lstm : {pd.DataFrame(y_pred_labels).value_counts()}')

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred_labels)
precision = precision_score(y_test, y_pred_labels, average='macro')  # 'macro' averages metrics across all classes
recall = recall_score(y_test, y_pred_labels, average='macro')
f1 = f1_score(y_test, y_pred_labels, average='macro')

# Display results as a table
results = {
        'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'],
        'Score': [accuracy, precision, recall, f1]
}
# Create DataFrame for easy display
results_df = pd.DataFrame(results)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_labels)
print("✅ \nConfusion Matrix:")
print(conf_matrix)
# Create classification report (including precision, recall, f1-score per class)
class_report = classification_report(y_test, y_pred_labels, target_names=['Class 0', 'Class 1'])
print("✅ \nClassification Report:")
print(class_report)

print(f"✅ Metrics Table:{results_df}")

############################################
 ########  Save LSTM  trained model  ########
 ############################################


#model.save("./models/binary_lstm.h5")
#print(f"✅  LSTM model saved")


#####################################################
################### Evaluating LSTM  predictor_07 ########
#####################################################

input_path = "./processed_data"
data_pred_file_name = "prediction_data.csv"
data_pred_file_path = os.path.join(input_path, data_pred_file_name)
data_pred = pd.read_csv(data_pred_file_path)
X_pred = data_pred.drop(columns = ['y'])


# Merging Class 1 and 2 together
y_pred = data.y.replace({3: 2})
y_pred = y_pred -1
print("✅ Unique y:", np.unique(y))
print(f"✅ Prediction Data separated (X,y- 2 Classes)")


X_pred_enc = np.expand_dims(X_pred, axis = 2)  # to be able to enter the LSTM model
y_pred_probs = model.predict(X_pred_enc)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

predictions = []
for label in y_pred_labels:
    if label == 0:
        prediction_text = "Your EEG is predicted to be tumor-induced seizure EEG."
        predictions.append(prediction_text)
    else:
        prediction_text = "Your EEG is predicted to be tumor/healthy baseline EEG."
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


# # #Basic LSTM Andy
# input_shape = (178,1)
# model = Sequential()

# model.add(LSTM(128, activation='tanh', return_sequences=True, input_shape=input_shape))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))

# model.add(LSTM(128, activation='tanh', return_sequences=True))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))


# model.add(LSTM(64, activation='tanh'))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))

# model.add(Dense(64, activation='relu'))
# model.add(Dense(3, activation='softmax'))

# print("✅ LSTM Model initialized")

# learning_rate = 0.001
# optimizer = Adam(learning_rate=learning_rate) # 'rmsprop'
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

# print("✅ LSTM Model compiled")

# # First Bidirectional LSTM layer with L2 regularization
# input_shape = (178,1)
# model = Sequential()
# model.add(Bidirectional(LSTM(128, activation='tanh',
#                             return_sequences=True,
#                             kernel_regularizer='l2'),
#                             input_shape=input_shape)
#           )

# model.add(BatchNormalization())
# model.add(Dropout(0.3))

# # Second Bidirectional LSTM layer with L2 regularization
# model.add(Bidirectional(LSTM(128, activation='tanh',
#                              return_sequences=True,
#                             kernel_regularizer='l2')))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))

# # Third LSTM layer with fewer units and L2 regularization
# model.add(LSTM(64, activation='tanh', kernel_regularizer='l2'))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))

# # Dense layer with ReLU activation
# model.add(Dense(64, activation='relu'))

# # Output layer with softmax activation for multi-class classification
# model.add(Dense(3, activation='softmax'))

# # Compile the model with the Adam optimizer and sparse categorical cross-entropy loss
# model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# print("✅ Improved LSTM Model1 initialized")
