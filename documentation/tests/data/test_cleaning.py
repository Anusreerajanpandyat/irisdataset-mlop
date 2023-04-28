import sys
sys.path.append('..')

import os
import sys

# Get the path of src directory
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))

# Add the path of src directory to the PYTHONPATH
sys.path.append(src_dir)



import pandas as pd
from src.data.cleaning import clean_data
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
X = iris['data']
y = iris['target']
data = pd.concat([X, y.rename('label')], axis=1)

def test_clean_data():
    # Create test data with missing values
    data_with_missing = data.copy()
    data_with_missing.iloc[0, 0] = pd.NA

    # Test that clean_data removes missing values
    cleaned_data = clean_data(data_with_missing)
    assert pd.isna(cleaned_data).sum().sum() == 0