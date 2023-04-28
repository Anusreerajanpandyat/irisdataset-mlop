import pandas as pd
from sklearn.model_selection import train_test_split
file_path = 'C:/Users/243415/Documents/irisdataset/documentation/data/raw/Iris.csv'
def load_data(filepath):
    """
    Load data from csv file and split it into training and testing sets.
    """
    data = pd.read_csv(filepath)
    X = data.drop(columns=['Species'])
    y = data['Species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test
