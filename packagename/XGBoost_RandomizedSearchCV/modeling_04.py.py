from sklearn.model_selection import RandomizedSearchCV

def perform_random_search(xgb, X_train, y_train, X_test, y_test):
    """
    Performs RandomizedSearchCV with the defined hyperparameter search space.
    """
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

    random_search.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=True
    )

    return random_search
