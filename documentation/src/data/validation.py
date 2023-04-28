from sklearn.metrics import accuracy_score

def validate_model(model, X_test, y_test):
    # Validate a model using accuracy score
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
