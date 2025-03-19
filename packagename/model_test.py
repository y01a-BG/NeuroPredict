import os
from pathlib import Path


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import LSTM, Dense, Dropout
from keras import callbacks
from keras.callbacks import EarlyStopping

print('Just the beginning...')
data = pd.read_csv("/home/jovana/code/y01a-BG/NeuroPredict/raw_data/Epileptic Seizure Recognition.csv")
print(data.head())

############################################
######## PREPROCESS ####################
#######################################


# (1) getting rid of not-needed column
data.drop(columns=['Unnamed'], inplace=True)

#(2) Taking only cats (1,2,3)
data = data[~data['y'].isin([4, 5])]


##############################################
############### DATA - gettin ready for LSTM ##########
##########################################################

X = data.drop(columns = 'y')
X_pad = np.expand_dims(X, axis=2)

y = data.y
y = y - 1  # Shift labels to start from 0 One-Hot-Encoder rule
y = to_categorical(y, num_classes=3)

# Split the data into 80% train and 20% test, with random_state=42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)



##################################################################
######################## LSTM (RNN) model #######################
####################################################################


###### (1)   Initialize the model ####################
model = Sequential()


model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(178,1)))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(64, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(3, activation='softmax'))

print(model.summary())


###### (2)   Compile the model ####################

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


###### (3)   FIT- Train the model with early Stopping####################
es = EarlyStopping(patience=1,
                   verbose=1,
                   restore_best_weights=True)

history = model.fit(X_train, y_train,
                     validation_split=0.2,
                    batch_size=128,
                    epochs=150,
                     callbacks=[es],
                   verbose = 1)


#####  (4) Predict on X_test ###############

y_pred = model.predict(X_test)

print(y_pred)
