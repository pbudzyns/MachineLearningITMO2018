import numpy as np
from sklearn.feature_selection import chi2 as _chi2
from sklearn.feature_selection import f_classif as _f_classif
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr


def chi2(X, y):
    return _chi2(X, y)


def f_classif(X, y):
    y = np.ndarray.flatten(y)
    # y = np.reshape(y, (1, max(m, n)))
    return _f_classif(X, y)


def mutual(x, y):
    return mutual_info_classif(x, y), None


def anova():
    pass


def pearson(x, y):
    y = np.ndarray.flatten(y)
    m, n = np.shape(x)
    scores = np.zeros((n, 2))
    for i in range(n):
        scores[i] = pearsonr(x[:, i], y)
    return scores[:, 0], scores[:, 1]
