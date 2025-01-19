import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from Models.Tensorflow import Tensorflow



# Importuje dane z dysku i zamienia dane na binarne
def importData(filename, columns):
    data = pd.read_csv(filename, names=columns, na_values="?")

    # Konwersja 'Outcome' na wartości binarne
    data['Outcome'] = data['Outcome'].map({'R': 1, 'N': 0})
    # Zamienia wartości NaN na 0
    data.fillna(0, inplace=True)

    return data


# Wybiera 10 najbardziej zkorelowanych cech
def select_top_features(data):
    # Obliczenie macierzy korelacji i wybór cech
    correlation_matrix = data.corr()
    correlation_with_outcome = correlation_matrix['Outcome'].abs().sort_values(ascending=False)
    top_features = correlation_with_outcome.index[1:11]  # Wybieramy 10 cech


    return top_features


# Zwraca zbiory treningowe i testowe
def export_sets(data, top_features):
    # Ograniczenie danych do wybranych cech
    X = data[top_features]
    print(X)
    y = data['Outcome']
    print(y)

    # Normalizacja cech
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    print(X)

    # Podział zbioru na treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # Utworzenie nazw kolumn do danych
    columns = ['ID', 'Outcome', 'Time', 'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness',
               'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry', 'Mean Fractal Dimension',
               'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE', 'Smoothness SE', 'Compactness SE', 'Concavity SE',
               'Concave Points SE', 'Symmetry SE', 'Fractal Dimension SE', 'Worst Radius', 'Worst Texture',
               'Worst Perimeter', 'Worst Area', 'Worst Smoothness', 'Worst Compactness', 'Worst Concavity',
               'Worst Concave Points', 'Worst Symmetry', 'Worst Fractal Dimension', 'Tumor Size', 'Lymph Node Status']

    # Podanie nazwy pliku
    filename = 'wpbc.data'

    data = importData(filename, columns)
    top_features = select_top_features(data)
    X_train, X_test, y_train, y_test = export_sets(data, top_features)
    model = Tensorflow(X_train, X_test, y_train, y_test)
    model.buildModel()


