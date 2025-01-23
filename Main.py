import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, Normalizer, StandardScaler, RobustScaler
from keras._tf_keras.keras.models import Sequential
from Models.RandomForestClassifier import RandomForestClassifierModel
from Models.BaggingClassifierModel import BaggingClassifierModel
from Models.BoostingClassifierModel import BoostingClassifierModel
from Models.Tensorflow import Tensorflow
from Models.BayesClassifierModel import BayesClassifierModel
from Models.SVMClassifier import SVMClassifier
from Models.DecisionTreeClassifier import DecisionTreeClassifierModel
from Models.XGBoostClassifier import XGBoostClassifier
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

def build_classificator(model):
    acc, report = model.buildModel()
    accuracies[model.name] = acc
    classification_reports[model.name] = report
    models[model.name] = model.model

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
    classification_reports = {}
    models = {
    "BaggingClassifier": BaggingClassifierModel,
    "AdaBoostClassifier": BoostingClassifierModel,
    "DecisionTreeClassifier": DecisionTreeClassifierModel,
    "BayesClassifier": BayesClassifierModel,
    "SVM": SVMClassifier,
    "XGBoostClassifier": XGBoostClassifier,
    "RandomForestClassifier": RandomForestClassifierModel
    }
    parameters = {
        # Ogólne parametry dla modeli Ensemble (Bagging, RandomForest, XGBoost, AdaBoost)
        "n_estimators": 200,  # Liczba estymatorów
        "max_depth": 15,  # Maksymalna głębokość drzewa
        "min_samples_split": 4,  # Minimalna liczba próbek do podziału
        "min_samples_leaf": 2,  # Minimalna liczba próbek w liściu

        "bootstrap": True,  # Próbkowanie z powtórzeniami (Bagging/RandomForest)
        "learning_rate": 0.05,  # Tempo uczenia (Boosting)
        "subsample": 0.8,  # Frakcja próbek używanych do trenowania każdego estymatora (XGBoost)
        "colsample_bytree": 0.8,  # Frakcja cech używanych w każdym drzewie (XGBoost)
        "gamma": 0.1,  # Regularyzacja dla podziałów (XGBoost)
        "random_state": 42,  # Losowość dla powtarzalności

        # Specyficzne dla SVM
        "C": 1.5,  # Współczynnik regularizacji
        "kernel": "rbf",  # Jądro radialne

        # Specyficzne dla DecisionTreeClassifier
        "criterion": "gini",  # Funkcja używana do pomiaru jakości podziału

        # Specyficzne dla BaggingClassifier
        "max_samples": 0.8,  # Proporcja próbek do losowania
    }
    for model in models.values():
        model = model(X_train, X_test, y_train, y_test)
        model.set_model_specs(**parameters)
        build_classificator(model)
    # Osobno dla Tensorflow
    model = Tensorflow(X_train, X_test, y_train, y_test)
    model.buildModel()
    plt = Plots()
    plt.plot_accuracies(accuracies)
    plt.plot_metrics(accuracies, classification_reports)
    for name, model in models.items():
        plt.plot_confusion_matrix(model, X_test, y_test, f'{name} Confusion Matrix')
    plt.plot_roc_curve(models, X_test, y_test)




