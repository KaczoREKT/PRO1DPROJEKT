import pandas as pd
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout
from keras._tf_keras.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Utworzenie nazw kolumn do danych
columns = ['ID', 'Outcome', 'Time', 'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness',
           'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry', 'Mean Fractal Dimension',
           'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE', 'Smoothness SE', 'Compactness SE', 'Concavity SE',
           'Concave Points SE', 'Symmetry SE', 'Fractal Dimension SE', 'Worst Radius', 'Worst Texture',
           'Worst Perimeter', 'Worst Area', 'Worst Smoothness', 'Worst Compactness', 'Worst Concavity',
           'Worst Concave Points', 'Worst Symmetry', 'Worst Fractal Dimension', 'Tumor Size', 'Lymph Node Status']

# Wczytanie pliku .data z dynamicznie zaimportowanymi nazwami kolumn
data = pd.read_csv('wpbc.data', names=columns, na_values="?")
print(data.head())

# Konwersja 'Outcome' na wartości binarne
data['Outcome'] = data['Outcome'].map({'R': 1, 'N': 0})

print(data['Outcome'].value_counts())

# Obliczenie macierzy korelacji i wybór cech
correlation_matrix = data.corr()
correlation_with_outcome = correlation_matrix['Outcome'].abs().sort_values(ascending=False)
top_features = correlation_with_outcome.index[1:11]  # Wybieramy 10 cech
print(f"Najbardziej skorelowane cechy z 'Outcome': {top_features.tolist()}")

# Ograniczenie danych do wybranych cech
X = data[top_features]
print(X)
y = data['Outcome']
print(y)

# Normalizacja cech
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Podział zbioru na treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Definicja modelu
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Pierwsza warstwa ukryta
    Dropout(0.3),  # Regularizacja, aby uniknąć przeuczenia
    Dense(32, activation='relu'),  # Druga warstwa ukryta
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Warstwa wyjściowa (sigmoid dla binarnej klasyfikacji)
])

# Kompilacja modelu
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',  # Funkcja strat dla klasyfikacji binarnej
              metrics=['accuracy']
              )

# Trenowanie modelu
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=16,
                    validation_data=(X_test, y_test),
                    verbose=1,
                    class_weight={0: 1, 1: 10})

# Ewaluacja modelu
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Dokładność na zbiorze testowym: {accuracy * 100:.2f}%")

# Predykcja i metryki
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(f"Dokładność (accuracy_score): {accuracy_score(y_test, y_pred) * 100:.2f}%")

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Dokładność (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność na zbiorze testowym: {accuracy * 100:.2f}%")

# Macierz pomyłek
cm = confusion_matrix(y_test, y_pred)
print("Macierz pomyłek:")
print(cm)

# Szczegółowy raport klasyfikacji
report = classification_report(y_test, y_pred)
print("Raport klasyfikacji:")
print(report)

actual_classes = []
predicted_classes = []
# Wyświetlenie rzeczywistej i przewidywanej klasy
for i, (true, pred) in enumerate(zip(y_test, y_pred.flatten())):
    actual_classes.append(true)
    predicted_classes.append(pred)

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Macierz pomyłek
cm = confusion_matrix(actual_classes, predicted_classes)

# Wizualizacja macierzy pomyłek
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Klasa 0', 'Klasa 1'], yticklabels=['Klasa 0', 'Klasa 1'])
plt.title('Macierz pomyłek')
plt.xlabel('Przewidywana klasa')
plt.ylabel('Rzeczywista klasa')
plt.show()

