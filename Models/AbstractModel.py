from abc import ABC
from sklearn.metrics import accuracy_score, classification_report

class AbstractModel(ABC):
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        self.name = None
        self.specs = {}
    def buildModel(self):
        self.model.fit(self.X_train, self.y_train)

        # Predykcje na zbiorze testowym
        y_pred = self.model.predict(self.X_test)

        # Wyświetlenie wyników
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"{self.name}\nAccuracy: {accuracy:.2f}")
        report = classification_report(self.y_test, y_pred, output_dict=True)
        print(f"Classification Report: {report}")

        return accuracy, report

    def set_model_specs(self, **kwargs):
        self.specs = kwargs
        valid_params = self.model.get_params().keys()
        filtered_parameters = {k: v for k, v in self.specs.items() if k in valid_params}
        self.model.set_params(**filtered_parameters)

