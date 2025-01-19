from Models.AbstractModel import AbstractModel
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, accuracy_score

class DecisionTreeClassifierModel(AbstractModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.model = DecisionTreeClassifier()

    def buildModel(self):
        # Trenowanie modelu drzewa decyzyjnego
        self.model.fit(self.X_train, self.y_train)

        # Predykcje na zbiorze testowym
        y_pred = self.model.predict(self.X_test)

        # Wyświetlenie wyników
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))

        # # Wyświetlenie reguł decyzyjnych
        # tree_rules = export_text(self.model, feature_names=[str(f) for f in range(self.X_train.shape[1])])
        # print("Decision Rules:")
        # print(tree_rules)

        return accuracy