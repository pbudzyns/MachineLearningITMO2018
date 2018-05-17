from sklearn.svm import SVC
import numpy as np
from random import randint
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


class SVM:
    """
        Simplified version if SMO algorithm for training SVMs.
    """

    def __init__(self):
        self.data = None
        self.targets = None
        self.kernel = None
        self.b = None
        self.alphas = None
        self.w = None
        self.sigma = None
        self.confusions = {'TP': 0, 'TN': 0, 'FP': 0, 'FT': 0}

    def fit(self, X, y, C=1, kernel='gaussian', tol=0.00001, max_iter=20, sigma=0.1):
        self.data = X.copy()
        self.targets = y.copy()
        self.kernel = kernel
        self.sigma = sigma

        m, n = np.shape(self.data)

        # Change 0 labels to -1
        self.targets[self.targets == 0] = -1
        self.targets = np.reshape(self.targets, (len(self.targets), 1))

        # Variables
        self.alphas = np.zeros((m, 1))
        self.b = 0
        E = np.zeros((m, 1))
        passes = 0

        lh_coutn, eta_count, old_alph = 0, 0, 0
        iterations = 0
        passed_if = 0

        # Pre-computing kernel martix
        K = self._apply_kernel()
        while passes < max_iter:
            num_changed_alphas = 0

            for i in range(m):
                iterations += 1

                # Calculate Ei = f(x(i) - y(i))
                # print('targets: ', np.shape(self.targets))
                # print('alphas: ', np.shape(self.alphas))
                # print('K: ', np.shape(K[:, i]))

                # tmp = np.multiply(self.alphas, self.targets)
                # tmp = self.alphas * self.targets * k
                # print(np.shape(tmp))


                # assert False
                k = np.reshape(K[:, i][:], (len(K[:, i]), 1))
                tmp = self.b + np.sum(self.alphas * self.targets * k) - self.targets[i]
                E[i] = tmp.copy()

                # print(np.shape(self.alphas[i]))
                # assert False
                # print('IF')
                # print(self.targets[i]*E[i], ' < ',  -tol, '=', self.targets[i]*E[i] < -tol)
                # print(self.alphas[i] , '< ',  C, '=', self.alphas[i] < C)
                # print(self.targets[i]*E[i], ' > ',  tol, '=', self.targets[i]*E[i] > tol)
                # print(self.alphas[i], ' > ',  0, ' = ', self.alphas[i] > 0)
                if (self.targets[i]*E[i] < -tol and self.alphas[i] < C)\
                        or (self.targets[i]*E[i] > tol and self.alphas[i] > 0):
                # if True:
                    passed_if += 1
                    # Randomly select j index and avoid same indexes
                    j = randint(0, m-1)
                    while j == i:
                        j = randint(0, m-1)

                    # Calculate Ej = f(x(j) - y(j))
                    k = np.reshape(K[:, j][:], (len(K[:, j]), 1))
                    tmp = self.b + np.sum(self.alphas * self.targets * k) - self.targets[j]
                    E[j] = tmp.copy()

                    alpha_i_old = self.alphas[i].copy()
                    alpha_j_old = self.alphas[j].copy()
                    # print('old i: ', alpha_i_old)
                    # print('old j: ', alpha_j_old)

                    if self.targets[i] == self.targets[j]:
                        tmp = (self.alphas[j].copy() + self.alphas[i].copy())
                        L = max(0, tmp - C)
                        H = min(C, tmp)
                        # print('i=j, L=', L, ' H=', H)
                    else:
                        tmp = (self.alphas[j].copy() - self.alphas[i].copy())
                        # print('tmp: ', tmp)
                        L = max(0, tmp)
                        H = min(C, C + tmp)
                        # print('i!=j, L=', L, ' H=', H)

                    if H == L:
                        # lh_coutn += 1
                        continue

                    eta = 2 * K[i, j].copy() - K[i, i].copy() - K[j, j].copy()

                    if eta >= 0:
                        # print('eta: ', eta)
                        eta_count += 1
                        continue

                    tmp = self.alphas[j].copy() - (self.targets[j].copy() * (E[i].copy() - E[j].copy())) / eta
                    # print(self.alphas[j], self.targets[j], E[i], E[j], eta)
                    # print('H ', H, 'L ', L, 'tmp ', tmp)
                    self.alphas[j] = tmp.copy()

                    self.alphas[j] = min(H, self.alphas[j].copy())
                    self.alphas[j] = max(L, self.alphas[j].copy())

                    # print('alha current, old', self.alphas[j], alpha_j_old)
                    if abs(self.alphas[j] - alpha_j_old) < tol:
                        self.alphas[j] = alpha_j_old.copy()
                        # print('Quiting: alpha_j_old')
                        # old_alph += 1
                        continue


                    tmp = self.alphas[i].copy() + self.targets[i]*self.targets[j]*(alpha_j_old - self.alphas[j])
                    self.alphas[i] = tmp

                    tmp = self.b - E[i].copy()\
                        - self.targets[j].copy() * (self.alphas[i].copy() - alpha_i_old) * K[i, j].copy()\
                        - self.targets[j].copy() * (self.alphas[j].copy() - alpha_j_old) * K[i, j].copy()
                    b1 = tmp.copy()
                    tmp = self.b - E[j]\
                        - self.targets[i].copy() * (self.alphas[i].copy() - alpha_i_old).copy() * K[i, j].copy()\
                        - self.targets[j].copy() * (self.alphas[j].copy() - alpha_j_old).copy() * K[j, j].copy()
                    b2 = tmp.copy()

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
        self.w = np.dot((self.alphas*self.targets).T, self.data)
        self.K = K

    def predict(self, X):
        # TODO: coplete predicting function
        m, n = np.shape(X)
        l, k = np.shape(self.data)
        predictions = np.zeros((m, 1))
        p = np.zeros((m, 1))
        if self.kernel == 'linear':
            p = np.dot(X, self.w.T) + self.b
        elif self.kernel == 'gaussian':
            for i in range(m):
                pred = 0
                for j in range(l):
                    pred = pred \
                           + self.alphas[j].copy() * self.targets[j]\
                           * self._gausian_kernel(X[i, :], self.data[j, :])
                p[i] = float(pred.copy() + self.b.copy())

        predictions[p >= 0] = 1
        predictions[p < 0] = 0
        return np.ndarray.flatten(predictions)

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

    def visualize_boundary(self, points, labels, model=None):
        self._plot_points(points, labels)

        model = self if not model else model

        # print('xx', np.min(points[:, 0]), np.max(points[:, 0]))
        # print('yy', np.min(points[:, 1]), np.max(points[:, 1]))

        x = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 100)
        y = np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), 100)

        xx, yy = np.meshgrid(x, y)
        values = np.zeros(np.shape(xx))
        from sklearn.svm import SVC
        # model = SVC(C=1.0, kernel='rbf')
        # model.fit(points, labels)

        for i in range(np.size(xx, 1)):
            tmp_x = np.c_[xx[:, i], yy[:, i]]
            values[:, i] = model.predict(tmp_x)

            # print(values[:, i])
            # assert False
            # values[:, i] = model.predict(tmp_x)
            # print('MY: ', self.predict(tmp_x))
            # print('SKL: ', model.predict(tmp_x))


        plt.contour(xx, yy, values)

        plt.show()

    def _apply_kernel(self):
        K = None
        if self.kernel == 'linear':
            K = np.dot(self.data, self.data.T)
        # elif self.kernel == 'gaussian':
        #     x2 = np.sum(self.data**2, axis=1)
        #     K = x2 + (x2.T - 2*np.dot(self.data, self.data.T))
        #     K = self._gausian_kernel(1, 0, sigma=2) ** K
        elif self.kernel == 'gaussian':
            m, n = np.shape(self.data)
            K = np.zeros((m, m))
            for i in range(m):
                for j in range(i, m):
                    K[i, j] = self._gausian_kernel(self.data[i, :], self.data[j, :]).copy()
                    K[j, i] = K[i, j]
        return K

    def _gausian_kernel(self, x, y):
        k = np.exp(-np.sum((x-y)**2)/(2*self.sigma**2))
        return k

    def _plot_points(self, points, labels):
        markers = {1: 'o', 0: 'x'}
        colors = {1: 'g', 0: 'r'}
        for point, label in zip(points, labels):
            plt.scatter(*point, c=colors[label], marker=markers[label])


if __name__ == '__main__':
    model = SVC()
    model.fit()