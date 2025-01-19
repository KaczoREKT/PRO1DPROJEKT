from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from Models.AbstractModel import AbstractModel


class BaggingClassifierModel(AbstractModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)

    def buildModel(self):
        # Trenowanie modelu Bagging
        self.model.fit(self.X_train, self.y_train)

        # Predykcje na zbiorze testowym
        y_pred = self.model.predict(self.X_test)

        # Wyświetlenie wyników
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))

        return accuracy
