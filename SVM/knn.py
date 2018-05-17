from queue import PriorityQueue
from sklearn.metrics import f1_score
import numpy as np
from sklearn.model_selection import train_test_split


class KNN:

    def __init__(self):
        self.features = None
        self.targets = None
        self.kernel = None
        self.k = 0
        self._kernels = [self.exp, self.nok, self.myk]

    def fit(self, X, y, k_max=20, kernel=None):
        # self.features = X
        # self.targets = y
        self.kernel = self._get_kernel(kernel)
        best_k = 1
        bes_acc = 0

        train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2)
        for k in range(1, k_max+1):
            self.features = train_data
            self.targets = train_labels
            acc = self.score(test_data, test_labels)
            if acc > bes_acc:
                bes_acc = acc
                best_k = k

        self.k = best_k
        self.features = X
        self.targets = y

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

    def get_confusions(self, X, y):
        predictions = self.predict(X)
        confusions = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

        for prediction, target in zip(predictions, y):
            if prediction == 1:
                if target == 1:
                    confusions['TP'] += 1
                elif target == 0:
                    confusions['FP'] += 1
            elif prediction == 0:
                if target == 1:
                    confusions['FN'] += 1
                elif target == 0:
                    confusions['TN'] += 1

        return confusions

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
