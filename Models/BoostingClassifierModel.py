from Models.AbstractModel import AbstractModel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier

class BoostingClassifierModel(AbstractModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.model = AdaBoostClassifier(**self.specs, estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
        self.name = 'AdaBoostClassifier'
