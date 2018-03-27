from matplotlib import pyplot as plt
from queue import PriorityQueue
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
    plt.show()


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


def split_data(points, labels):
    from sklearn.model_selection import train_test_split
    # data = [(x, y, l) for x, y, l in zip(points, labels)]
    return train_test_split(points, labels, test_size=0.2)


def euclidean_distance(a_point, b_point):
    return ((a_point[0]-b_point[0])**2 + (a_point[1]-b_point[1])**2)**(1/2)


def null_kernel(distance):
    return distance


def simple_kernel(distance):
    return 1/(1+distance)


def knn(k, test_point, train_points, train_labels, dist_function, kernel):
    results = PriorityQueue()
    for point, label in zip(train_points, train_labels):
        dist = dist_function(point, test_point)
        results.put((dist, label))
    score = [0] * 2
    for i in range(k):
        value, cls = results.get()
        score[int(cls)] += 1
    # print(score)
    return np.argmax(score)


def test(knn, k, kernel, distance, points, labels, test_points, test_labels):

    true_positive = 0
    predicted_positive = 0
    actual_positive = sum(test_labels)

    # TODO: Check if Fscore implementation is correct

    for test_point, test_label in zip(test_points, test_labels):
        predicted_class = knn(k, test_point, points, labels, distance, kernel)
        if predicted_class == 1:
            predicted_positive += 1
        if predicted_class == 1 and test_label == 1:
            true_positive += 1

        if predicted_class == 1:
            if test_label == 1:
                plt.scatter(test_point[0], test_point[1], marker='o', color='y')
            else:
                plt.scatter(test_point[0], test_point[1], marker='x', color='k')

        elif predicted_class == 0:
            if test_label == 0:
                plt.scatter(test_point[0], test_point[1], marker='x', color='y')
            else:
                plt.scatter(test_point[0], test_point[1], marker='o', color='k')


    #plot_data(points, labels)

    Fscore = round(get_f_score(true_positive, predicted_positive, actual_positive), 2)

    result = "{} | {} | Neighbours number: {} | Result: {}".format(distance.__name__, kernel.__name__, k, Fscore)
    print(result)

    return Fscore, result


def get_f_score(true_positive, predicted_positive, actual_positive):
    try:
        precision = true_positive/predicted_positive
        recall = true_positive/actual_positive
    except ZeroDivisionError:
        return 0
    return 2*(precision*recall)/(precision+recall)


if __name__ == "__main__":

    # Load raw data from file
    points, labels = read_data('chips.txt')

    # Plot points with their labels
    plot_data(points, labels)

    # Split data for learning and testing set and show them
    points, test_points, labels, test_labels = split_data(points, labels)
    plot_splited_data(points, labels, test_points, test_labels)
    #print(test_labels, test_points)

    # result = knn(20, test_points[0], points, labels, euclidean_distance, null_kernel)
    # print(result)
    # result = knn(20, test_points[0], points, labels, euclidean_distance, simple_kernel)
    # print(result)
    best_F = 0
    best_result = ''

    for metric in [euclidean_distance]:
        for kernel in [simple_kernel]:
            for k in range(1, 21):
                F1, result = test(knn, k, kernel, metric, points, labels, test_points, test_labels)
                best_F, best_result = (F1, result) if F1 > best_F else (best_F, best_result)

    print("-----------------------------------------------------")
    print("\nBest F-score: {}\n".format(best_F), best_result)