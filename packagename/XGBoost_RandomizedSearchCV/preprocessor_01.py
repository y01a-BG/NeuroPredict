import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)
    data.drop(columns=['Unnamed'], inplace=True)
    data = data[data['y'].isin([1, 2, 3])]

    # Data preparation: extract features and adjust labels to start from 0
    X = data.drop(columns=['y']).values
    y = data['y'].values - 1  # Now labels are 0, 1, 2

    # Splitting dataset with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Feature scaling for optimal performance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
