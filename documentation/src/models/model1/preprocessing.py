from sklearn.preprocessing import StandardScaler

def preprocess_data(X_train, X_test):
    """
    Standardize the data using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    return X_train_std, X_test_std

