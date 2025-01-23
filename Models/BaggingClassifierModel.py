from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from Models.AbstractModel import AbstractModel


class BaggingClassifierModel(AbstractModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.model = BaggingClassifier(**self.specs, estimator=DecisionTreeClassifier())
        self.name = 'BaggingClassifier'

