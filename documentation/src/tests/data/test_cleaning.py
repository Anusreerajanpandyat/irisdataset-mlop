import pandas as pd
from data.cleaning import clean_data


def test_clean_data():
    # Create test data with missing values
    data_with_missing = data.copy()
    data_with_missing.iloc[0, 0] = pd.NA

    # Test that clean_data removes missing values
    cleaned_data = clean_data(data_with_missing)
    assert pd.isna(cleaned_data).sum().sum() == 0