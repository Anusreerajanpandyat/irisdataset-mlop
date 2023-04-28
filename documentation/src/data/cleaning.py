import pandas as pd

def clean_data(data):
    # Remove any rows with missing data
    data = data.dropna()
    return data
