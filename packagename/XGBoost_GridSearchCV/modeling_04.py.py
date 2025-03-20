from sklearn.model_selection import GridSearchCV

def perform_grid_search(xgb, X_train, y_train, X_test, y_test):
    """
    Performs grid search using early stopping.
    """
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="mlogloss",
        early_stopping_rounds=10,
        verbose=True
    )

    return grid_search
