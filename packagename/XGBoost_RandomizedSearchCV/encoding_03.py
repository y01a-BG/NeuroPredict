from xgboost import XGBClassifier

def create_xgb_model():
    """
    Returns a configured XGBClassifier with early_stopping_rounds and eval_metric.
    """
    xgb = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        verbosity=1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss',
        early_stopping_rounds=10
    )
    return xgb
