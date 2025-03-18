
from preprocessor import clean_data
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras import callbacks
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


data = pd.read_csv("../raw_data/Epileptic Seizure Recognition.csv")
data = clean_data(data)

X = data.drop(columns = 'y')
X = np.expand_dims(X, axis=2)  # to be able to enter the LSTM model
y = data.y


# classification on y-column
y = y - 1  # Shift labels to start from 0
y = to_categorical(y, num_classes=3)

# Split the data into 80% train and 20% test, with random_state=42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#####################################################
################### MODEL LSTM ######################
#####################################################

# input_shape = (178,1)
def initialize_model(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """

    model = Sequential()
    model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(178,1)))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(LSTM(64, activation='relu'))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(3, activation='softmax'))

    print("✅ Model initialized")

    return model

model = initialize_model((178,1))

def compile_model(model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate) # 'rmsprop'
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

    print("✅ Model compiled")

    return model


model = compile_model(model, learning_rate=0.0005)
def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=128,
        patience=1,
        validation_data=None, # overrides validation_split
        validation_split=0.2
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=150,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )

    print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history


model, history = train_model(model,
                            np.array(X_train),np.array(y_train),
                            batch_size=128,
                            patience=1,
                            validation_data=None, # overrides validation_split
                            validation_split=0.2
                            )

'''
def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    mae = metrics["mae"]

    print(f"✅ Model evaluated, MAE: {round(mae, 2)}")

    return metrics
'''
