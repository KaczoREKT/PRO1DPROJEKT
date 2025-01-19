import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, Normalizer, StandardScaler, RobustScaler
from Models.BaggingClassifierModel import BaggingClassifierModel
from Models.BoostingClassifierModel import BoostingClassifierModel
from Models.Tensorflow import Tensorflow
from Models.IntercontinentalBayes import BayesClassifier
from Models.SVMClassifier import SVMClassifier
from Models.DecisionTreeClassifier import DecisionTreeClassifierModel
from Plots import Plots


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
    og_X = data[top_features]
    og_y = data['Outcome']

    # Lista technik normalizacji
    scalerList = [MinMaxScaler, StandardScaler, RobustScaler, Normalizer, MaxAbsScaler]

    # Wybór modelu
    index = 1

    # Normalizacja danych
    scaler = scalerList[index]()
    X = scaler.fit_transform(og_X)

    # Podział zbioru na treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, og_y, test_size=0.25, random_state=42)

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

    # Mapa dokładności każdego modelu
    accuracies = {}

    # Test klasyfikatora TensorFlow
    print("Test klasyfikatora Tensorflow")
    model = Tensorflow(X_train, X_test, y_train, y_test)
    accuracies['TensorFlow'] = model.buildModel()

    # Test klasyfikatora NaiveBayes
    print("Test klasyfikatora NaiveBayes")
    model = BayesClassifier(X_train, X_test, y_train, y_test)
    accuracies['BayesClassifier'] = model.buildModel()

    # Test klasyfikatora SVM
    print("Test klasyfikatora SVM")
    model = SVMClassifier(X_train, X_test, y_train, y_test)
    accuracies['SVM'] = model.buildModel()

    # Test klasyfikatora DecisionTreeClassifier
    print("Test klasyfikatora DecisionTreeClassifier")
    model = DecisionTreeClassifierModel(X_train, X_test, y_train, y_test)
    accuracies['Decision Tree'] = model.buildModel()

    # Test klasyfikatora BaggingClassifierModel
    print("Test klasyfikatora BaggingClassifierModel")
    model = BaggingClassifierModel(X_train, X_test, y_train, y_test)
    accuracies['Bagging'] = model.buildModel()

    # Test klasyfikatora BoostingClassifierModel
    print("Test klasyfikatora BoostingClassifierModel")
    model = BoostingClassifierModel(X_train, X_test, y_train, y_test)
    accuracies['Boosting'] = model.buildModel()
    plt = Plots()
    plt.plot_accuracies(accuracies)
    plt.plot_metrics(accuracies)


