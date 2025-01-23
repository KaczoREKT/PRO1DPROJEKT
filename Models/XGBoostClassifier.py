from xgboost import XGBClassifier

from Models.AbstractModel import AbstractModel


class XGBoostClassifier(AbstractModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.name = "XGBoostClassifier"
        self.model = XGBClassifier(**self.specs, random_state=42)
