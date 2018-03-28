from matplotlib import pyplot as plt
from queue import PriorityQueue
from sklearn.model_selection import train_test_split
import sklearn as skl
import numpy as np
import csv


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
        dist = kernel(point, test_point)
        results.put((dist, label))
    score = [0] * 2
    for i in range(k):
        value, cls = results.get()
        score[int(cls)] += 1
    return np.argmax(score)


def test_knn(k, kernel, points, labels, test_points, test_labels):

    true_positive = 0
    predicted_positive = 0
    actual_positive = sum(test_labels)
    predicted_labels = []
    correctly_predicted = 0

    # TODO: Check if Fscore implementation is correct
    for test_point, test_label in zip(test_points, test_labels):
        predicted_class = knn(k, test_point, points, labels, kernel)
        if predicted_class == 1:
            predicted_positive += 1
        if predicted_class == 1 and test_label == 1:
            true_positive += 1
        if predicted_class == test_label:
            correctly_predicted += 1

        predicted_labels.append(predicted_class)

    #Fscore = round(get_f_score(true_positive, predicted_positive, actual_positive), 2)
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
    print('\tq - exit')


def get_k_and_kernel():
    k = int(input('Neighbor number: '))
    kernel = input('Kernel: ')
    return k, kernel


def euc(x, y):
    return ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** (1 / 2)


def lin(x, y):
    val = skl.metrics.pairwise.linear_kernel([x], [y])[0][0]
    return val


def gaus(x, y):
    val = skl.metrics.pairwise.sigmoid_kernel([x], [y], gamma=0.1, coef0=0)[0][0]
    return val


def get_kernel(kernel_name):
    kernel = None
    for func in [euc, lin, gaus]:
        if kernel_name == func.__name__:
            kernel = func
            break
    return kernel


def main(filename):
    base_points, base_labels = read_data(filename)
    points, labels = base_points[:], base_labels[:]
    test_points, test_labels = [], []
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
            find_best(points, test_points, labels, test_labels, k_max, kernel)
        elif opt == 's':
            percent = float(input('Test data percent: '))
            points, test_points, labels, test_labels = split_data(base_points, base_labels, percent)
        elif opt == 'm':
            kernel = None
            while not kernel:
                kernel_name = input('K: ')
                kernel = get_kernel(kernel_name)
            print(kernel(points[0], points[1]))
        elif opt == 'q':
            break


def find_best(points, test_points, labels, test_labels, k_max, kernel):
    best_F = -1
    best_result = ''
    for k in range(1, k_max):
        _p, F1, result = test_knn(k, kernel, points, labels, test_points, test_labels)
        best_F, best_result = (F1, result) if F1 > best_F else (best_F, best_result)

    print(best_result)
    return best_F, best_result


if __name__ == "__main__":

    main(filename='chips.txt')
