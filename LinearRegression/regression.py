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

    with open(filename, 'r') as f:
        data = []
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        for value in reader:
            data.append([int(x) for x in value])

        dataArray = np.array(data)
        features = dataArray[:,:2]
        prizes = dataArray[:,-1]

        N = np.shape(features)[0]
        features2 = np.c_[np.ones([N, 1]), features]

    return np.array(features2), prizes


def getThetas(n):
    a = 1/(2*n)
    thetas = [random.random()*2*a-a for i in range(n)]
    return np.array(thetas)


def normalize(data):
    mean = np.mean(data)
    sigma = np.std(data)
    return (data - mean)/sigma, mean, sigma

def normalize_features(features):
    n, m = np.shape(features)
    sigma = []
    mu = []
    _features = []
    for i in range(1, m):
        data, _mu, _sigma = normalize(features[:, i])
        _features.append(data)
        sigma.append(_sigma)
        mu.append(_mu)

    #print(np.shape(np.array(_features).T))
    features_normalized = np.c_[np.ones([n, 1]), np.array(_features).T]
    #print(np.shape(features_normalized))
    return features_normalized, mu, sigma


def regression(features, thetas):
    return np.dot(features,thetas)


def getError(prediction, real_values):
    return sum(prediction - real_values)


def gradientDescent(features, thetas, results, alpha=0.01,  maxIter=1000, maxError=1):
    resultThetas = thetas[:]
    errors = []
    #print(np.dot(features, thetas))
    #for i in range(nIter):
    #    resultThetas, e = gradientDescentStep(features, resultThetas, results, alpha)
    #    errors.append(e)
    resultThetas, error = gradientDescentStep(features, resultThetas, results, alpha)

    while abs(error) > maxError and maxIter:
        resultThetas, error = gradientDescentStep(features, resultThetas, results, alpha)
        errors.append(abs(error))
        maxIter -= 1
        print(resultThetas)

    return resultThetas, errors


def gradientDescentStep(features, thetas, results, alpha=0.01):
    resultThetas = thetas[:]
    m = len(features)
    #print(features*thetas)
    #print('RESULTS:\n\n', np.dot(features, thetas))
    y = np.dot(features, thetas)# - results
    error = np.dot(y, y.T)/(2/m)
    dTheta = alpha/m * np.dot(features.T, np.dot(features, thetas)-results)
    resultThetas -= dTheta
    #print(error)
    return resultThetas, error


def prediction(area, rooms, thetas, mu, sigma):
    _x = (area - mu[0])/sigma[0]
    _y = (rooms - mu[1])/sigma[1]
    return thetas[0]+thetas[1]*_x+thetas[2]*_y


def plotData(features, prizes):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(features[:,0], features[:,1], prizes)

    ax.set_xlabel('area')
    ax.set_ylabel('rooms')
    ax.set_zlabel('prize')

    plt.show()

def plotDataAfterLearning(features, prizes, thetas, errors):
    fig = plt.figure()
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(features[:,1], features[:,2], prizes)

    x1, x2 = np.min(features[:,1]), np.max(features[:,1])
    y1, y2 = np.min(features[:,2]), np.max(features[:,2])

    xx, yy = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    ax2=fig.add_subplot(132, projection='3d')
    ax2.scatter(features[:,1], features[:,2], prizes)
    ax2.plot_surface(xx, yy, thetas[2]*xx+thetas[1]*yy)

    ax3 = fig.add_subplot(133)
    ax3.plot(errors)
    plt.show()

if __name__ == "__main__":
    features, prizes = load_data('prices.txt')

    #plotData(features, prizes)
    thetas = getThetas(np.shape(features)[1])
    #thetas = np.zeros(np.shape(features)[1])

    features_normalized, mu, sigma = normalize_features(features)
    #print(np.shape(features_normalized))
    #print(np.dot(features_normalized,thetas) - prizes)
    thetas, errors = gradientDescent(features_normalized, thetas, prizes, 0.001, 1500)

    plotDataAfterLearning(features_normalized, prizes, thetas, errors)
    print(prediction(2104, 3, thetas, mu, sigma))

