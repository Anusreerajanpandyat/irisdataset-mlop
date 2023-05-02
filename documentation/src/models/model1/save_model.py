import os
import pickle
from train import clf, model_path

model_path = 'C:/Users/243415/Documents/irisdataset/documentation/models/svm_model.pkl'

# Save the trained model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
with open(model_path, "wb") as f:
    pickle.dump(clf, f)