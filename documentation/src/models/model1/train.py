import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataloader import load_data
from preprocessing import preprocess_data
from model import SVMClassifier
import pickle

# Set random seed for reproducibility
random_seed = 42

# Set paths for data 

data_path ='Iris.csv'


# Load the data
X_train, X_test, y_train, y_test = load_data(data_path)

# Preprocess the data
X_train, X_test= preprocess_data(X_train, X_test)

# Standardize the feature data
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Train the model
clf = SVMClassifier()
clf.train(X_train_std, y_train)

# Evaluate the model
accuracy = clf.clf.score(X_test_std, y_test)
print(f"Accuracy: {accuracy:.2f}")



