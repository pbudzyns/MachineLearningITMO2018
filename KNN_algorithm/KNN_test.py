from matplotlib import pyplot as plt
from queue import PriorityQueue
from sklearn.model_selection import train_test_split
import sklearn as skl
import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier
from KNN_algorithm.cross_validation import CrossValidator


def read_data(filename):
    data = list()
    labels = list()
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            data.append((float(row[0]), float(row[1])))
            labels.append(int(row[2]))

    return np.array(data), np.array(labels)


def plot_data(points, labels=None):
    for point, label in zip(points, labels):
        if label == 1:
            plt.scatter(point[0], point[1], marker='o', c='g')
        elif label == 0:
            plt.scatter(point[0], point[1], marker='x', c='r')


def plot_splited_data(points, labels, test_points, test_labels):
    for point, label in zip(points, labels):
        if label == 1:
            plt.scatter(point[0], point[1], marker='o', c='g')
        elif label == 0:
            plt.scatter(point[0], point[1], marker='x', c='r')

    for point, label in zip(test_points, test_labels):
        if label == 1:
            plt.scatter(point[0], point[1], marker='o', c='b')
        elif label == 0:
            plt.scatter(point[0], point[1], marker='x', c='b')

    plt.show()


def plot_after_training(points, test_points, labels, test_labels, predicted_labels):
    plot_data(points, labels)

    for test_point, test_label, predicted_label in zip(test_points, test_labels, predicted_labels):
        if predicted_label == 1:
            if test_label == 1:
                plt.scatter(test_point[0], test_point[1], marker='o', color='y')
            else:
                plt.scatter(test_point[0], test_point[1], marker='x', color='k')

        elif predicted_label == 0:
            if test_label == 0:
                plt.scatter(test_point[0], test_point[1], marker='x', color='y')
            else:
                plt.scatter(test_point[0], test_point[1], marker='o', color='k')

    plt.show()


def split_data(points, labels, percent):
    return train_test_split(points, labels, test_size=percent)





def null_kernel(distance):
    return distance


def simple_kernel(distance):
    return 1/(1+distance)


def test_and_plot(k, kernel, points, test_points, labels, test_labels):
    predicted_labels, best_f, result = test_knn(k, kernel, points, labels, test_points, test_labels)
    print(result)
    plot_after_training(points, test_points, labels, test_labels, predicted_labels)


def knn(k, test_point, train_points, train_labels, kernel):
    results = PriorityQueue()
    for point, label in zip(train_points, train_labels):
        # dist = kernel(point, test_point)
        dist = euc(point, test_point)
        score = dist
        results.put((score, label))
    score = [0] * 2
    for i in range(k):
        value, cls = results.get()
        score[int(cls)] += kernel(value)
    return np.argmax(score)


def test_knn(k, kernel, points, labels, test_points, test_labels):

    true_positive = 0
    predicted_positive = 0
    actual_positive = sum(test_labels)
    predicted_labels = []
    correctly_predicted = 0

    for test_point, test_label in zip(test_points, test_labels):
        predicted_class = knn(k, test_point, points, labels, kernel)
        if predicted_class == 1:
            predicted_positive += 1
        if predicted_class == 1 and test_label == 1:
            true_positive += 1
        if predicted_class == test_label:
            correctly_predicted += 1

        predicted_labels.append(predicted_class)

    # Fscore = round(get_f_score(true_positive, predicted_positive, actual_positive), 2)
    Fscore = round(correctly_predicted/len(test_labels), 2)

    result = " {} | Neighbours number: {} | Result: {}".format(kernel.__name__, k, Fscore)

    return predicted_labels, Fscore, result


def get_f_score(true_positive, predicted_positive, actual_positive):
    try:
        precision = true_positive/predicted_positive
        recall = true_positive/actual_positive
    except ZeroDivisionError:
        return 0
    return 2*(precision*recall)/(precision+recall)


def printMenu():
    print('-------------\tKNN training\t-----------------')
    print('Possible kernels: [ ]')
    print('\ts - split data')
    print('\tt - start training')
    print('\tf - find best')
    print('\tm - test krenels')
    print('\tk - cros validation for given NN')
    print('\tq - exit')


def get_k_and_kernel():
    k = int(input('Neighbor number: '))
    kernel = input('Kernel [lin, gaus, myk, poli, nok]: ')
    return k, kernel


def euc(x, y):
    return ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** (1 / 2)


def lin(x, y=None):
    val = skl.metrics.pairwise.linear_kernel(x, y)[0][0]
    return val


def myk(x, y=None):
    return 1/x


def exp(x, y=None):
    return np.exp(-(x-8))


def poli(x, y=None):
    val = skl.metrics.pairwise.polynomial_kernel(x, y)[0][0]
    # val = 1/(x**2)
    return val


def gaus(x, y=None):
    # val = skl.metrics.pairwise.sigmoid_kernel([x], [y], gamma=0.1, coef0=0)[0][0]
    val = skl.metrics.pairwise.sigmoid_kernel(x, y, gamma=None, coef0=0.2)[0][0]
    return val


def nok(x, y=None):
    return 1


def get_kernel(kernel_name):
    kernel = None

    for func in [lin, gaus, myk, poli, nok]:
        if kernel_name == func.__name__:
            return func
    return kernel


def test_kernels(dist):
    for func in [lin, gaus, myk, poli, nok]:
        print(func.__name__, ': ', func(dist))


def find_best(points, test_points, labels, test_labels, k_max, kernel):
    best_F = -1
    best_result = ''
    for k in range(1, k_max):
        _p, F1, result = test_knn(k, kernel, points, labels, test_points, test_labels)
        best_F, best_result = (F1, result) if F1 > best_F else (best_F, best_result)

    print(best_result)
    return best_F, best_result


def main(filename):
    base_points, base_labels = read_data(filename)
    points, labels = base_points[:], base_labels[:]
    crsVal = CrossValidator(points, labels)
    crsVal.splitDataIntoParts(10)
    test_points, test_labels = [], []
    kernels_list = [nok, myk, exp]
    printMenu()
    while True:
        opt = input('What to do: ')
        if opt == 't':
            k, kernel_name = get_k_and_kernel()
            kernel = get_kernel(kernel_name)
            test_and_plot(k, kernel, points, test_points, labels, test_labels)
        elif opt == 'f':
            k_max, kernel_name = get_k_and_kernel()
            kernel = get_kernel(kernel_name)
            for func in kernels_list:
                find_best(points, test_points, labels, test_labels, k_max, func)
        elif opt == 's':
            percent = float(input('Test data percent: '))
            points, test_points, labels, test_labels = split_data(base_points, base_labels, percent)

        elif opt == 'g':
            k_max, kernel_name = get_k_and_kernel()
            kernel = get_kernel(kernel_name)
            for k in range(1, k_max+1):
                predicted_labels, best_f, result = test_knn(k, kernel, points, labels, test_points, test_labels)
                print(result)

        elif opt == 'k':
            k_max = int(input('Neighbors number: '))



            for kernel in kernels_list:
                F_scores = []
                for points, test_points, labels, test_labels in crsVal.getValidationPart():
                    _, F1, result = test_knn(k_max, kernel, points, labels, test_points, test_labels)
                    F_scores.append(F1)

                print("{} | NN: {} | Score: {}".format(kernel.__name__, k_max, np.mean(F_scores)))


        elif opt == 'h':
            k = int(input('Neighbor number: '))
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(points, labels)

            print('sklearn res: ', neigh.score(test_points, test_labels))

        elif opt == 'm':
            print('Point1: ', points[0])
            print('Point2: ', points[1])
            dist = euc(points[0], points[1])
            print('Euc dist: ', dist)
            test_kernels(dist)
            # kernel = None
            # while not kernel:
            #     kernel_name = input('K: ')
            #     kernel = get_kernel(kernel_name)
            # print(kernel(points[0], points[1]))
        elif opt == 'q':
            break


if __name__ == "__main__":

    main(filename='chips.txt')
    # print(gaus(5))
    # print(lin(5))