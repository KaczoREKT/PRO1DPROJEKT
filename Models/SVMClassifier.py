from Models.AbstractModel import AbstractModel
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC

class SVMClassifier(AbstractModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.model = SVC(kernel='linear', probability=True)

    def buildModel(self):
        # Trenowanie modelu SVM
        self.model.fit(self.X_train, self.y_train)

        # Predykcje na zbiorze testowym
        y_pred = self.model.predict(self.X_test)

        # Wyświetlenie wyników
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))

        return accuracy