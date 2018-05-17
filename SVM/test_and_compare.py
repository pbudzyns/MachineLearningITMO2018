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


def cross_validation():
    knn = KNN()
    svm = SVM()

    knn_scores, knn_fscores = [], []
    svm_scores, svm_fscores = [], []

    # Cross validation
    for train_points, test_points, train_labels, test_labels in crsValidation.iterate():
        knn.fit(train_points, train_labels, kernel='exp')
        svm.fit(train_points, train_labels, C=1.0, kernel='gaussian', tol=0.001, sigma=0.1)

        knn_scores.append(knn.score(test_points, test_labels))
        knn_fscores.append(knn.f_score(test_points, test_labels))
        svm_scores.append(svm.score(test_points, test_labels))
        svm_fscores.append(svm.f_score(test_points, test_labels))

    print('Average values: ')
    print('KNN: accuracy: ', np.mean(knn_scores), ', f-score: ', np.mean(knn_fscores))
    print('SVM: accuracy: ', np.mean(svm_scores), ', f-score: ', np.mean(svm_fscores))


def svm_train_with_visualization():
    svm = SVM()
    svm.fit(points, labels, C=1.0, kernel='gaussian', tol=0.001, sigma=0.3)
    svm.visualize_boundary(points, labels)


def compare_using_wilcoxon():
    knn = KNN()
    svm = SVM()

    knn_scores, knn_fscores = [], []
    svm_scores, svm_fscores = [], []

    # from scipy.stats import wilcoxon as wcx
    # Cross validation
    for train_points, test_points, train_labels, test_labels in crsValidation.iterate():
        knn.fit(train_points, train_labels, kernel='exp')
        svm.fit(train_points, train_labels, C=1.2, kernel='gaussian', tol=0.001, sigma=0.1)

        knn_predictions = knn.predict(test_points)
        svm_predictions = svm.predict(test_points)

        # print(wcx(knn_predictions, svm_predictions))
        stat, p = wilcoxon(knn_predictions, svm_predictions)
        print('W: ', stat, ' p: ', p)


if __name__ == '__main__':

    # compare_to_library()

    points, labels = read_data('chips.txt')
    crsValidation = CrossValidation(points[:], labels[:])
    crsValidation.splitDataIntoParts(10)

    # svm_train_with_visualization()
    #
    # cross_validation()
    #
    compare_using_wilcoxon()


    # train_points, test_points, train_labels, test_labels = crsValidation.iterate().__next__()
    # knn.fit(train_points, train_labels, k=6, kernel='exp')
    # from sklearn.svm import SVC
    # model = SVC()
    # model.fit(train_points, train_labels)
    # svm.fit(points, labels, C=1, kernel='gaussian', max_iter=20, tol=0.001)
    # svm.fit(train_points, train_labels, C=1, kernel='gaussian', max_iter=20, tol=0.001, sigma=0.2)
    # print(svm.w)
    # print(svm.alphas)
    # print(svm.K[:, 0])
    # print(svm._gausian_kernel(points[0], points[1]))

    # print(svm.predict(train_points))
    # print(svm.score(test_points, test_labels))
    # print(svm.f_score(test_points, test_labels))

    # print(model.score(test_points, test_labels))

    # svm.visualize_boundary(train_points, train_labels)
    # svm.visualize_boundary(train_points, train_labels, model=model)
    # knn_predictions = knn.predict(test_points)
    # svm_predictions = svm.predict(test_points)
    #
    # res_knn = wilcoxon(knn_predictions, svm_predictions)
    # print(res_knn)

    # res_svm = wilcoxon(svm_predictions, test_labels)
