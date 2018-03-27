import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import random
import os

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
        prices = dataArray[:,-1]

        N = np.shape(features)[0]
        # Add ones column
        features2 = np.c_[np.ones([N, 1]), features]

    return np.array(features2), prices


def getThetas(n):
    # Initialize theta with random values
    a = 1/(2*n)
    thetas = [random.random()*2*a-a for i in range(n)]
    return np.array(thetas)


def normalize(data):
    mean = np.mean(data)
    sigma = np.std(data)
    return (data - mean)/sigma, mean, sigma


def normalize_features(features):
    n, m = np.shape(features)
    sigmas = []
    means = []
    _features = []

    for i in range(1, m):
        data, _mu, _sigma = normalize(features[:, i])
        _features.append(data)
        sigmas.append(_sigma)
        means.append(_mu)

    # Transform result lists into one array with ones column
    features_normalized = np.c_[np.ones([n, 1]), np.array(_features).T]
    return features_normalized, means, sigmas


def gradientDescent(features, thetas, results, alpha=0.01,  maxIter=1000, maxError=1):
    resultThetas = thetas[:]
    errors = []
    resultThetas, error = gradientDescentStep(features, resultThetas, results, alpha)

    while abs(error) > maxError and maxIter:
        resultThetas, error = gradientDescentStep(features, resultThetas, results, alpha)
        errors.append(abs(error))
        maxIter -= 1

    return resultThetas, errors


def gradientDescentStep(features, thetas, results, alpha=0.01):
    resultThetas = thetas[:]
    m = len(features)
    #print(features*thetas)
    #print('RESULTS:\n\n', np.dot(features, thetas))
    y = np.dot(features, thetas) - results
    #error = np.dot(y, y.T)#/(2/m)
    error = sum(y)
    #dTheta = (alpha/m) * np.dot(features.T, np.dot(features, thetas)-results)
    dTheta = (alpha) * np.dot(features.T, np.dot(features, thetas)-results)
    resultThetas -= dTheta
    #print(error)
    return resultThetas, error


def makePrediction(area, rooms, thetas, mu, sigma):
    _x = (area - mu[0])/sigma[0]
    _y = (rooms - mu[1])/sigma[1]
    return thetas[0]+thetas[1]*_x+thetas[2]*_y


def plotData(features, prices):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(features[:,0], features[:,1], prices)

    ax.set_xlabel('area')
    ax.set_ylabel('rooms')
    ax.set_zlabel('prize')

    plt.show()


def plotDataAfterLearning(features, means, sigmas, prices, thetas, errors):
    # TODO: add titles to plots
    fig = plt.figure()
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(features[:,1], features[:,2], prices)
    ax.set_xlabel('area')
    ax.set_ylabel('rooms')
    ax.set_zlabel('prize')
    ax.set_title('Given data')

    x1, x2 = np.min(features[:, 1]), np.max(features[:, 1])
    y1, y2 = np.min(features[:, 2]), np.max(features[:, 2])+1

    xx, yy = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    ax2=fig.add_subplot(132, projection='3d')
    ax2.scatter(features[:, 1], features[:, 2], prices)
    ax2.plot_surface(xx, yy, thetas[2]*(xx - means[0])/sigmas[0] + thetas[1]*(yy - means[1])/sigmas[1] + thetas[0],
                     color='y')
    ax2.set_xlabel('area')
    ax2.set_ylabel('rooms')
    ax2.set_zlabel('prize')
    ax2.set_title('Data with result plane')

    ax3 = fig.add_subplot(133)
    ax3.plot(errors)
    ax3.set_title('Errors')
    plt.show()


def printMenu():
    print('+++++++++\tLinear regression\t+++++++++++')
    print('\tt - start training')
    print('\tp - show plots')
    print('\tk - make prediction')
    print('\tq - quit\n')


def startTraining(features, prices):
    thetas = getThetas(np.shape(features)[1])
    features_normalized, mu, sigma = normalize_features(features)
    thetas, errors = gradientDescent(features_normalized, thetas, prices, 0.01, 350)

    return thetas, mu, sigma, errors


def main(data_filename):
    printMenu()
    features, prices = load_data(data_filename)
    mu, sigma, thetas, errors = [], [], [], []
    while True:
        try:
            opt = input('What to do: ')
            if opt == 't':
                thetas, mu, sigma, errors = startTraining(features, prices)
                print('Training finished...')
                print('Final equation: %.2f*area + %.2f*rooms + %.2f'%(thetas[1], thetas[2], thetas[0]))
            elif opt == 'p':
                plotDataAfterLearning(features, mu, sigma, prices, thetas, errors)
            elif opt == 'k':
                area = int(input('Area: '))
                rooms = int(input('Rooms: '))
                prize = round(makePrediction(area, rooms, thetas, mu, sigma), 2)
                print('Predicted prize: ', prize)
            elif opt == 'q':
                break
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main('prices.txt')



