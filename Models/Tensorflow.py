from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout
from keras._tf_keras.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report

from Models.AbstractModel import AbstractModel


class Tensorflow(AbstractModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(self.X_train.shape[1],)),  # Pierwsza warstwa ukryta
            Dropout(0.4),  # Regularizacja, aby uniknąć przeuczenia
            Dense(64, activation='relu'),  # Druga warstwa ukryta
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')  # Warstwa wyjściowa (sigmoid dla binarnej klasyfikacji)
        ])
        self.name = 'Tensorflow'
        self.parameters = {}

    def buildModel(self):
        # Kompilacja modelu
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                           loss='binary_crossentropy',  # Funkcja strat dla klasyfikacji binarnej
                           metrics=['accuracy']
                           )
        # Trenowanie modelu
        self.model.fit(self.X_train, self.y_train,
                       epochs=150,
                       batch_size=32,
                       validation_data=(self.X_test, self.y_test),
                       verbose=1,
                       class_weight={0: 1, 1: 3})

        # Ewaluacja modelu
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Dokładność na zbiorze testowym: {accuracy * 100:.2f}%")

        # Predykcja i metryki
        y_pred = (self.model.predict(self.X_test) > 0.5).astype(int)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Dokładność (accuracy_score): {accuracy_score(self.y_test, y_pred) * 100:.2f}%")
        report = classification_report(self.y_test, y_pred, output_dict=True)
        print(f"Classification Report: {report}")

        return accuracy, report
