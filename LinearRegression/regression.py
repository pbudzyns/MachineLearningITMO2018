import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

"""Second task. Linear regression homework.
It is required to retrieve the coefficients of linear regression by one of the two ways: 
gradient descent or genetic algorithm. You can't use existing implementations from libraries.
The choice of hyperparameters and the configuration method is left for you, but be prepared 
to answer additional questions on them. The dataset is the dependence of the cost of housing on the area and the
number of rooms. Use mean squared error as an empirical risk.
Your program must have an ability to get additional input points 
(e.g., from console) for checking the already trained model."""


def load_data(filename):

    features = []
    prizes = []

    with open(filename, 'r') as f:
        data = []
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        for value in reader:
            data.append([int(x) for x in value])

        dataArray = np.array(data)
        features = dataArray[:,:2]
        prizes = dataArray[:,-1]

    return features, prizes


def getThetas(n):
    thetas = []
    a = 1/(2*n)
    for i in range(n):
        value = random.random()*2*a - a
        thetas.append(value)
    return np.array(thetas)

def regression(features, thetas):
    return np.dot(features,thetas)


def getError(prediction, real_values):
    return sum(prediction - real_values)


def gradientDescent():
    pass


def plotData(features, prizes):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(features[:,0], features[:,1], prizes)

    ax.set_xlabel('area')
    ax.set_ylabel('rooms')
    ax.set_zlabel('prize')

    plt.show()


if __name__ == "__main__":
    features, prizes = load_data('prices.txt')

    plotData(features, prizes)
    thetas = getThetas(np.shape(features)[1])
    predictions = regression(features, thetas)
    prizes = prizes/1000
    print(getError(predictions, prizes))

