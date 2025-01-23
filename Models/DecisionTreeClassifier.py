from Models.AbstractModel import AbstractModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

class DecisionTreeClassifierModel(AbstractModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.model = DecisionTreeClassifier(**self.specs)
        self.name = "DecisionTreeClassifier"


