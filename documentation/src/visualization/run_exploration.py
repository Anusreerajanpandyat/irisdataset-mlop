# run_exploration.py

import pandas as pd
from exploration import create_histogram, create_scatterplot, create_heatmap,create_pairplot


file_path = 'C:/Users/243415/Documents/irisdataset/documentation/data/raw/Iris.csv'
iris=pd.read_csv(file_path)

# Create histogram of sepal length
create_histogram(iris, 'SepalLengthCm')

# Create scatterplot of petal length vs. petal width, colored by species
create_scatterplot(iris, 'PetalLengthCm', 'PetalWidthCm', 'Species')

# Create heatmap of correlation matrix
create_heatmap(iris)



#pairplot
create_pairplot(iris, 'Species')
