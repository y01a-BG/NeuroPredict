import pandas as pd
import numpy as np



import tensorflow as tf
from keras.models import Sequential

from keras.layers import LSTM, Dense, Dropout,BatchNormalization
from keras.optimizers import Adam




#####################################################
################### MODEL LSTM ######################
#####################################################

# input_shape = (178,1)
def initialize_model(input_shape: tuple):
    """
    Initialize the Neural Network with random weights
    """

    model = Sequential()
    model.add(LSTM(128, activation='tanh', return_sequences=True, input_shape=(178,1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(LSTM(128, activation='tanh', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))


    model.add(LSTM(64, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    print("✅ Model initialized")

    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the Neural Network
    """
    optimizer = Adam(learning_rate=learning_rate) # 'rmsprop'
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

    print("✅ Model compiled")

    return model

####################################
#####################################

'''
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
