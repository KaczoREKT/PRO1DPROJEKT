class Classifier:
    def __init__(self, X_train, x_test, y_train, y_test):
        self.X_train = X_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None

    def change_model(self, model):
        self.model = model

