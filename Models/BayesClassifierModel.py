from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

from Models.AbstractModel import AbstractModel


class BayesClassifierModel(AbstractModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.model = GaussianNB(**self.specs)
        self.name = 'BayesClassifier'

