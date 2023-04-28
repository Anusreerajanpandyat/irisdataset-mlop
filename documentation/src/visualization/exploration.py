# exploration.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'C:/Users/243415/Documents/irisdataset/documentation/data/raw/Iris.csv'
iris=pd.read_csv(file_path)

def create_histogram(df, feature):
    """Creates a histogram of the specified feature."""
    plt.hist(df[feature])
    plt.title(f"{feature} distribution")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.show()
    
def create_scatterplot(df, x_feature, y_feature, target):
    """Creates a scatterplot of the specified x and y features, colored by target variable."""
    sns.scatterplot(data=df, x=x_feature, y=y_feature, hue=target)
    plt.title(f"{x_feature} vs {y_feature}")
    plt.show()
    
def create_heatmap(df):
    """Creates a heatmap of the correlation matrix."""
    df_numeric = df.select_dtypes(include=['float64', 'int64'])  # select only numeric columns
    corr_matrix = df_numeric.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation matrix")
    plt.show()

def create_pairplot(df, target):
    """Creates a pairplot of all features, colored by target variable."""
    sns.pairplot(data=df, hue=target)
    plt.show()
