from SVM.svm import SVM
from SVM.knn import KNN
from SVM.cross_validation import CrossValidation
from scipy.stats import distributions

import numpy as np
import csv

"""
It is required to write SVM algorithm without using any libraries and count the f-measure and the confusion matrix
using the developed model.
Next, you need investigate what Wilcoxon test is, use this test to compare kNN and SVM algorithms
and calculate the p-value. You should use old dataset chips.txt (from the first lab).
You should be able to answer any theory questions about SVM and about the statistical test.
"""


def read_data(filename):
    data = list()
    labels = list()
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            data.append((float(row[0]), float(row[1])))
            labels.append(int(row[2]))

    return np.array(data), np.array(labels)


def sgn(x):
    return 1 if x>=0 else -1


def compare_to_library():
    from sklearn.svm import SVC

    c = 5

    points, labels = read_data('chips.txt')
    crsValidation = CrossValidation(points[:], labels[:])
    crsValidation.splitDataIntoParts(118)

    svm_my = SVM()
    svm_skl = SVC(C=c, kernel='rbf')

    scores_skl = []
    scores_my = []

    for train_points, test_points, train_labels, test_labels in crsValidation.iterate():
        svm_my.fit(train_points, train_labels, C=c, max_iter=40)
        svm_skl.fit(train_points, train_labels)

        scores_skl.append(svm_skl.score(test_points, test_labels))
        scores_my.append(svm_my.score(test_points, test_labels))

    print('Accuracy:')
    print('Score by sklearn: ', round(np.mean(scores_skl), 3))
    print('Score by mine: ', round(np.mean(scores_my), 3))


def wilcoxon(x, y):
    from scipy.stats import rankdata
    d = x - y

    d = np.compress(np.not_equal(d, 0), d, axis=-1)
    n = len(d)

    # Computing statistic
    r = (len(d) + 1) * 0.5

    r_p = np.sum((d > 0) * r)
    r_n = np.sum((d < 0) * r)

    T = min(r_p, r_n)

    # Computing p-value
    p_value = 0

    ex = n * (n + 1) * 0.25
    var = n * (n + 1) * (2 * n + 1) / 24

    z = (T - ex)/np.sqrt(var)

    p_value = 2. * distributions.norm.sf(abs(z))

    return T, p_value


if __name__ == '__main__':

    # compare_to_library()

    points, labels = read_data('chips.txt')
    crsValidation = CrossValidation(points[:], labels[:])
    crsValidation.splitDataIntoParts(4)

    knn = KNN()
    svm = SVM()

    for train_points, test_points, train_labels, test_labels in crsValidation.iterate():
        knn.fit(train_points, train_labels, k=6, kernel='exp')
        svm.fit(train_points, train_labels, C=0.5, kernel='linear', max_iter=40)

        knn_predictions = knn.predict(test_points)
        svm_predictions = svm.predict(test_points)

        res_knn = wilcoxon(knn_predictions, svm_predictions)
        print(res_knn)

    # res_svm = wilcoxon(svm_predictions, test_labels)
