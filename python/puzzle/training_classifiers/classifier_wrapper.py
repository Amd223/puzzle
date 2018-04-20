from random import random

from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier


class ClassifierWrapper:

    def __init__(self, classifier):
        self.classifier = classifier

    def get_name(self):
        return self.classifier.__class__.__name__

    def fit(self, x_train, y_train):
        if isinstance(self.classifier, KNeighborsClassifier):
            no_samples = min(len(y_train), 100000)
            ids = list(range(no_samples))
            random.shuffle(ids)
            x_train_knn = [x_train[i] for i in ids]
            y_train_knn = [y_train[i] for i in ids]
            xs, ys = x_train_knn, y_train_knn
        else:
            xs, ys = x_train, y_train

        self.classifier.fit(xs, ys)

    def score(self, x_test, y_test):
        return self.classifier.score(x_test, y_test)

    def predict(self, x_test):
        if hasattr(self.classifier, "decision_function"):
            return self.classifier.decision_function(x_test)
        else:
            return self.classifier.predict_proba(x_test)[:, 1]

    def get_roc_curve(self, y, x_test):
        pred = self.predict(x_test)
        x, y = roc_curve(y, pred)
        area = auc(x, y)
        return x, y, area
