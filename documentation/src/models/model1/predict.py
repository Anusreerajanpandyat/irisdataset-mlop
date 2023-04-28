import pickle

model_path = "models/trained_model.pkl"

def predict(model_path, X_test_std):
    """
    Load the trained model and make predictions on new data.
    """
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
        
    y_pred = clf.predict(X_test_std)
    return y_pred
