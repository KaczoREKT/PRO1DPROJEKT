
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


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
                    class_weight={0: 1, 1: 5})

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

