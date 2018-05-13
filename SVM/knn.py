from queue import PriorityQueue
from sklearn.metrics import f1_score
import numpy as np


class KNN:

    def __init__(self):
        self.features = None
        self.targets = None
        self.kernel = None
        self.k = 2
        self._kernels = [self.exp, self.nok, self.myk]

    def fit(self, X, y, k=2, kernel=None):
        self.features = X
        self.targets = y
        self.k = k
        self.kernel = self._get_kernel(kernel)

    def predict(self, X):
        if np.ndim(X) > 1:
            result = []
            for x in X:
                result.append(self._predict(x))
        else:
            result = self._predict(X)
        return result

    def score(self, X, y):
        predictions = self.predict(X)
        correct_pred = 0
        for prediction, target in zip(predictions, y):
            if prediction == target:
                correct_pred += 1
        return correct_pred/len(y)

    def f_score(self, X, y):
        predictions = self.predict(X)
        return f1_score(y, predictions)

    def _predict(self, X):
        results = PriorityQueue()
        for point, label in zip(self.features, self.targets):
            dist = self._compute_distance(point, X)
            results.put((dist, label))
        score = [0] * 2
        for i in range(self.k):
            value, cls = results.get()
            score[int(cls)] += self.kernel(value)
        return np.argmax(score)

    def _compute_distance(self, x, y):
        return ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** (1 / 2)

    def _get_kernel(self, kernel):
        if not kernel:
            return self.nok
        for func in self._kernels:
            if func.__name__ == kernel:
                return func

    # Kernels
    def nok(self, x):
        return 1

    def exp(self, x, shift=10):
        return np.exp(-(x-shift))

    def myk(self, x):
        return 1/x
