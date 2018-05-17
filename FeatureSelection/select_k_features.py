import numpy as np
from FeatureSelection.metrics import chi2, mutual, pearson, f_classif
from queue import PriorityQueue
from sklearn.svm import SVC
np.warnings.filterwarnings('ignore')

from sklearn.feature_selection import SelectKBest

"""
You should implement the feature selection algorithm based on the utility metric (the Filter method).
Implement several utility metrics and compare their performance at classification tasks.
"""


class SelectKFeatures:

    def __init__(self, score_func, k):
        self.score_func = score_func
        self.k = k
        self._scores = None
        self._indexes = None
        self._pvalues = None

    def fit(self, features, labels):
        self._scores, self._pvalues = self.score_func(features, labels)
        self._indexes = self._get_k_best_indexes()
        self._mask = self._get_mask()

    def transform(self, X):
        # m, n = np.shape(features)
        # result = np.zeros((m, self.k))
        # for i, vector in enumerate(features):
        #     result[i-1] = vector[self._mask]
        # X = np.array(X)
        # mask = np.asarray(self._mask)
        # print(mask)
        # return X[:, mask]
        X = np.asarray(X)
        # print('Transform: scores from indexes', self._scores[self._indexes])
        return X[:, self._indexes]

    def _transform_vector(self, v):
        res = []
        for idx in self._indexes:
            res.append(v[idx])

        return res

    def _get_k_best_indexes(self):
        q = PriorityQueue()
        for idx, score in enumerate(self._scores):
            # q.put((-score, idx))
            q.put((1/score, idx))
            # if score > 2390: print('score: ', score, 'idx', idx-1)

        indexes = []
        for i in range(self.k):
            score, val = q.get()
            # print((score, val))
            indexes.append(val)
            # print('getting score: ', score, 'idx ', val)

        return sorted(indexes)

    def _get_mask(self):
        scores = self._scores
        mask = np.zeros(scores.shape, dtype=bool)
        mask[np.argsort(scores, kind="mergesort")[-self.k:]] = 1

        return mask


def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append([int(x) for x in f.readline().split()])
    return np.array(data)


if __name__ == '__main__':

    arcene_train_data = load_data('arcene/arcene_train.data')
    arcene_train_labels = load_data('arcene/arcene_train.labels')

    arcene_valid_data = load_data('arcene/arcene_valid.data')
    arcene_valid_labels = load_data('arcene/arcene_valid.labels')

    metrics = [chi2, f_classif, mutual, pearson]
    # metrics = [pearson]
    # featureFilter = SelectKFeatures(score_func=chi2, k=10)
    # featureFilter.fit(arcene_train_data, arcene_train_labels)
    # print(featureFilter.transform([arcene_train_data[0]]))
    #
    # test = SelectKBest(score_func=chi2, k=10)
    # test.fit(arcene_train_data, arcene_train_labels)
    # print(test.transform([arcene_train_data[0]]))

    for metric in metrics:
        featureFilter = SelectKFeatures(score_func=metric, k=6)
        featureFilter.fit(arcene_train_data, arcene_train_labels)

        new_features = featureFilter.transform(arcene_train_data)
        new_test = featureFilter.transform(arcene_valid_data)

        svm = SVC(C=1.0, kernel='rbf')
        svm.fit(new_features, arcene_train_labels)
        print(metric.__name__, '| accuracy:', svm.score(new_test, arcene_valid_labels),
              ' | selected features (indexes): ', featureFilter._indexes)

