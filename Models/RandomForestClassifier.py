from Models.AbstractModel import AbstractModel
from sklearn.ensemble import RandomForestClassifier

class RandomForestClassifierModel(AbstractModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.model = RandomForestClassifier(**self.specs)
        self.name = "RandomForestClassifier"


