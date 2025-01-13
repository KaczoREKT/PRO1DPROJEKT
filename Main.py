import pandas as pd

columns = ['ID', 'Outcome', 'Time', 'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness',
           'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry', 'Mean Fractal Dimension',
           'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE', 'Smoothness SE', 'Compactness SE', 'Concavity SE',
           'Concave Points SE', 'Symmetry SE', 'Fractal Dimension SE', 'Worst Radius', 'Worst Texture',
           'Worst Perimeter', 'Worst Area', 'Worst Smoothness', 'Worst Compactness', 'Worst Concavity',
           'Worst Concave Points', 'Worst Symmetry', 'Worst Fractal Dimension', 'Tumor Size', 'Lymph Node Status']

# Wczytanie pliku .data z dynamicznie zaimportowanymi nazwami kolumn
data = pd.read_csv('wpbc.data', names=columns)

# Podgląd danych
print(data.head())
