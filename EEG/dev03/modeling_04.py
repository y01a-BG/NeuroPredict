import pandas as pd
import numpy as np



import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, BatchNormalization, Dropout, Dense, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from xgboost import XGBClassifier
#from sklearn.model_selection import RandomizedSearchCV

from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier


#****************************************************************#
#****************************************************************#
#****************************************************************#
#   THIS .py CONTAINS MODELING FOR  DIFFERENT ML-models:
#   1. LSTM
#   2. XG Boost
#   3.RocketClassifier (from sktime)
#   4. ShapeletTransformClassifier (from sktime)
#****************************************************************#
#****************************************************************#
#****************************************************************#




#####################################################
################### MODEL LSTM ######################
#####################################################

#Bidirectional LSTM model
def initialize_lstm_model(input_shape = (178,1)):
    """
    Initialize the Neural Network with random weights
    """

    model = Sequential()

    # First Bidirectional LSTM layer with L2 regularization
    model.add(Bidirectional(LSTM(128, activation='tanh', return_sequences=True,
                                  kernel_regularizer='l2'), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Second Bidirectional LSTM layer with L2 regularization
    model.add(Bidirectional(LSTM(128, activation='tanh', return_sequences=True,
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
    ######  Above XGBoost best params are the result of randomSearchCV below #######
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



####################################################
#################  MODEL    Rocket ##################
######################################################

def initialize_rocket_model():
    clf = RocketClassifier(num_kernels=1000,
                        rocket_transform='rocket',
                        max_dilations_per_kernel=32,
                        n_features_per_kernel=4,
                        use_multivariate='auto',
                        n_jobs=-1,
                        random_state=None
    )

    return clf


########################################################################
#############      MODEL    ShapeletTransformClassifier  ###############
################ THIS MODEL TAKES ~ 4h on GPU  #########################
########################################################################

def initialize_shapelet_model():
    shapelet_class = ShapeletTransformClassifier(
        n_shapelet_samples=1000,
        max_shapelets=None, max_shapelet_length=None,
        estimator=None, transform_limit_in_minutes=0,
        time_limit_in_minutes=0,
        save_transformed_data=False,
        n_jobs=1, batch_size=100, random_state=None
        )

    return shapelet_class



#################################################
######### if runing: python  modelin_04.py ######
#################################################

def initialize_models(model: str):
    """
    This function will apply encoding based on the model choice (LSTM ,XGBoost...).

    Parameters:
    - X_train: Input training data
    - y_train: Target training data
    - model: A string indicating the model type ('LSTM', 'XGBoost', 'Rocket', 'Shapelet')

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
    elif model == "Rocket":
        rocket_model = initialize_rocket_model()
        print("✅ Rocket Model initialized")
        return rocket_model
    elif model == "Shapelet":
        shapelet_model = initialize_shapelet_model()
        print("✅ Shapelet Model initialized")
        return shapelet_model
    else:
        print(f"Unknown model: {model}. Please choose either: 'LSTM', 'XGBoost', 'Rocket', 'Shapelet'")


if __name__ == "__main__":
    #lstm_model = initialize_models('LSTM')
    xgb_model_out = initialize_models('XGBoost')
    #rocket_model_out = initialize_models('Rocket')
    #shapelet_model_out = initialize_models('Shapelet')
