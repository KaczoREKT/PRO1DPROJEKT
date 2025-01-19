from abc import ABC, abstractmethod

class AbstractModel(ABC):
    @abstractmethod
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    @abstractmethod
    def buildModel(self):
        pass