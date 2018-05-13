from sklearn.svm import SVC
import numpy as np
from random import randint
from sklearn.metrics import f1_score


class SVM:

    def __init__(self):
        self.data = None
        self.targets = None
        self.kernel = None
        self.b = None
        self.alphas = None
        self.w = None

    def fit(self, X, y, C=1, kernel='linear', tol=0.00001, max_iter=5):
        self.data = X[:]
        self.targets = y[:]
        self.kernel = kernel

        m, n = np.shape(self.data)
        self.targets[self.targets == 0] = -1

        self.alphas = np.zeros((m, 1))
        self.b = 0
        E = np.zeros((m, 1))
        passes = 0

        K = self._apply_kernel()
        while passes < max_iter:
            num_changed_alphas = 0

            for i in range(m):

                E[i] = self.b + np.sum(self.targets*self.alphas.T*K[:, i]) - self.targets[i]

                if (self.targets[i]*E[i] < -tol and self.alphas[i] < C)\
                        or (self.targets[i]*E[i] > tol and self.alphas[i] > 0):
                    j = randint(0, m-1)
                    while j == i:
                        j = randint(0, m-1)

                    # print('j: ', j)

                    E[j] = self.b + np.sum(self.targets*self.alphas.T*K[:, j]) - self.targets[j]

                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]

                    if self.targets[i] == self.targets[j]:
                        L = max(0, self.alphas[j] + self.alphas[i] - C)
                        H = min(C, self.alphas[j] + self.alphas[i])
                    else:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(C, C + self.alphas[j] - self.alphas[i])

                    if H == L:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    self.alphas[j] = self.alphas[j] - (self.targets[j] * (E[i] - E[j])) / eta

                    self.alphas[j] = min(H, self.alphas[j])
                    self.alphas[j] = max(L, self.alphas[j])

                    if abs(self.alphas[j] - alpha_j_old) < tol:
                        self.alphas[j] = alpha_j_old
                        # print('Quiting: alpha_j_old')
                        continue

                    self.alphas[i] = self.alphas[i] + self.targets[i]*self.targets[j]*(alpha_j_old - self.alphas[j])

                    b1 = self.b - E[i]\
                        - self.targets[j] * (self.alphas[i] - alpha_i_old) * K[i, j]\
                        - self.targets[j] * (self.alphas[j] - alpha_j_old) * K[i, j]

                    b2 = self.b - E[j]\
                        - self.targets[i] * (self.alphas[i] - alpha_i_old) * K[i, j]\
                        - self.targets[j] * (self.alphas[j] - alpha_j_old) * K[j, j]

                    if 0 < self.alphas[i] < C:
                        self.b = b1
                    elif 0 < self.alphas[j] < C:
                        self.b = b2
                    else:
                        self.b = (b1+b2)/2

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        idx = np.ndarray.flatten(self.alphas > 0)
        self.data = self.data[idx, :]
        self.targets = self.targets[idx]
        self.alphas = self.alphas[idx]
        self.w = np.dot((self.alphas.T*self.targets), self.data)

    def predict(self, X):
        # TODO: coplete predicting function
        predictions = np.zeros((len(X), 1))
        p = np.dot(X, self.w.T) + self.b
        predictions[p >= 0] = 1
        predictions[p < 0] = 0
        return predictions

    def score(self, X, y):
        # TODO: complete function for accuracy measuring
        predictions = self.predict(X)
        correct = 0
        for pred, target in zip(predictions, y):
            if pred == target:
                correct += 1
        return correct/len(y)

    def f_score(self, X, y):
        # TODO: complete function for f_score
        predictions = self.predict(X)

        return f1_score(y, predictions)

    def _apply_kernel(self):
        if self.kernel == 'linear':
            return np.dot(self.data, self.data.T)
        elif self.kernel == 'gaussian':
            x2 = np.sum(self.data**2, axis=1)
            return None
        elif self.kernel == 'centerTransform':
            return np.sum(self.data**2, axis=1)

if __name__ == '__main__':
    model = SVC()
    model.fit()