import pandas as pd
from sklearn.datasets import load_iris
import pytest


from data.cleaning import clean_data
from data.ingestion import load_data
from data.splitting import split_data
from data.validation import validate_model
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer

# Create test data
iris = load_iris(as_frame=True)
X = iris['data']
y = iris['target']
data = pd.concat([X, y.rename('label')], axis=1)

# Test clean_data function
def test_clean_data():
    # Create test data with missing values
    data_with_missing = data.copy()
    data_with_missing.iloc[0, 0] = pd.NA

    # Test that clean_data removes missing values
    cleaned_data = clean_data(data_with_missing)
    assert pd.isna(cleaned_data).sum().sum() == 0

# Test load_data function
def test_load_data():
    # Test that load_data returns a DataFrame
    assert isinstance(load_data(), pd.DataFrame)

# Test split_data function
def test_split_data():
    # Test that split_data returns the expected number of rows
    X_train, X_test, y_train, y_test = split_data(data, test_size=0.2, random_state=0)
    assert len(X_train) + len(X_test) == len(data)

# Test validate_model function
def test_validate_model():
    # Test that validate_model returns an accuracy score between 0 and 1
    # Create a simple SVM pipeline
    svm = make_pipeline(
        make_column_transformer((StandardScaler(), X.columns)),
        SVC(kernel='linear', C=1, random_state=0)
    )

    # Fit the pipeline on the iris data
    svm.fit(X, y)

    # Test the pipeline's accuracy on the iris data
    accuracy = validate_model(svm, X, y)
    assert 0 <= accuracy <= 1
