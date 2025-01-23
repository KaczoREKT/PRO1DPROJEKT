from Models.AbstractModel import AbstractModel
from sklearn.svm import SVC

class SVMClassifier(AbstractModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.model = SVC(**self.specs, kernel='linear', probability=True)
        self.name = 'SVM'

