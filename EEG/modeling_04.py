import pandas as pd
import numpy as np



import tensorflow as tf
from keras.models import Sequential

from keras.layers import LSTM, Dense, Dropout,BatchNormalization
from keras.optimizers import Adam


from xgboost import XGBClassifier
#from sklearn.model_selection import RandomizedSearchCV


#****************************************************************#
#****************************************************************#
#****************************************************************#
#   THIS .py CONTAINS MODELING FOR  DIFFERENT ML-models:
#   1. LSTM
#   2. XG Boost
#   3.
#****************************************************************#
#****************************************************************#
#****************************************************************#




#####################################################
################### MODEL LSTM ######################
#####################################################


def initialize_lstm_model(input_shape = (178,1) ):
    """
    Initialize the Neural Network with random weights
    """

    model = Sequential()
    model.add(LSTM(128, activation='tanh', return_sequences=True, input_shape=input_shape))
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

    print("✅ LSTM Model initialized")

    return model


def compile_lstm_model(model, learning_rate=0.001):
    """
    Compile the Neural Network
    """
    optimizer = Adam(learning_rate=learning_rate) # 'rmsprop'
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

    print("✅ LSTM Model compiled")

    return model

####################################################
#################  MODEL    XGBoost ##################
######################################################

def initialize_xgb_model():
    """
    Returns a configured XGBClassifier with early_stopping_rounds and eval_metric.
    Best Parameters from Randomised Search

    """

    best_params = {
        'subsample': 0.8,
        'reg_lambda': 1,
        'reg_alpha': 1.0,
        'n_estimators': 300,
        'max_depth': 9,
        'learning_rate': 0.05,
        'gamma': 0,
        'colsample_bytree': 0.8
        }


    xgb = XGBClassifier(
            subsample=best_params['subsample'],
            reg_lambda=best_params['reg_lambda'],
            reg_alpha=best_params['reg_alpha'],
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            gamma=best_params['gamma'],
            colsample_bytree=best_params['colsample_bytree'],

            objective='multi:softprob',
            eval_metric="mlogloss",
            num_class=3,
            verbosity=1,
            random_state=42,
            use_label_encoder=False
        )

    print("✅ XGBoost Model initialized")
    return xgb


    ################################################################################
    ##############   Above best params are the result of randomSearchCV below #######
    ################################################################################

    ''''
    param_distributions = {
        'n_estimators': [150, 200, 250, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [5, 7, 9],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.5],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [1, 1.5, 2]
        }


    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_distributions,
        n_iter=50,
        scoring='f1_macro',
        cv=3,
        n_jobs=-1,
        verbose=2,
        random_state=42
        )
    '''
#################################################
######### if runing: python  modelin_04.py ######
#################################################

def initialize_models(model: str):
    """
    This function will apply encoding based on the model choice (LSTM or XGBoost).

    Parameters:
    - X_train: Input training data
    - y_train: Target training data
    - model: A string indicating the model type ('LSTM' or 'XGBoost')

    Returns:
    - Encoded X_train and y_train based on the selected model.
    """

    if model == "LSTM":
        input_shape = (178,1)
        model = initialize_lstm_model(input_shape)
        lstm_model = compile_lstm_model(model)
        print("✅ LSTM Model initialized")
        print("✅ LSTM Model compiled")
        return lstm_model

    elif model == "XGBoost":
        xgb_model = initialize_xgb_model()
        print("✅ XGBoost Model initialized")
        return xgb_model
    else:
        print(f"Unknown model: {model}. Please choose either 'LSTM' or 'XGBoost'.")


if __name__ == "__main__":
    #lstm_model = initialize_models('LSTM')
    xgb_model_out = initialize_models('XGBoost')
