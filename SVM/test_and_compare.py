from SVM.svm import SVM
from SVM.knn import KNN
from SVM.cross_validation import CrossValidation

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

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



if __name__ == '__main__':

    points, labels = read_data('chips.txt')
    crsValidation = CrossValidation(points[:], labels[:])
    crsValidation.splitDataIntoParts(10)

    knn = KNN()
    svm = SVM()
    svm_skl = SVC(C=1.0, kernel='rbf')

    scores = []
    scores2 = []
    f_score = []

    for train_points, test_points, train_labels, test_labels in crsValidation.iterate():
        svm.fit(train_points, train_labels, C=1, max_iter=40)
        svm_skl.fit(train_points, train_labels)

        scores.append(svm_skl.score(test_points, test_labels))
        scores2.append(svm.score(test_points, test_labels))
        f_score.append(svm.f_score(test_points, test_labels))

    print('Score sklearn: ', np.mean(scores))
    print('Score my: ', np.mean(scores2))
    print('F_score my: ', np.mean(f_score))

    # TODO: Getting confusion matrix for SVM algorithm

    # TODO: Use Wilcoxon test to compare SVM and KNN and calculate p-value
