from xgboost import XGBClassifier

def create_xgb_model():
    """
    Returns a configured XGBClassifier for grid search.
    """
    xgb = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        verbosity=1,
        random_state=42,
        use_label_encoder=False
    )
    return xgb
