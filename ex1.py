
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def computeCost(X, y, theta):
    """
    Compute cost = ((x * theta) - y)^2
    price = 4
    poulation = 2
    theta (muss gefunden werden) 2

    ((2 * 2) - 4)^2 = 0
    """
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


def gradientDescent(X, y, theta, alpha, iters):
    tmp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            tmp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = tmp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

if __name__ == "__main__":
    # Read data
    path = os.path.join(os.getcwd() + '/data/ex1data1.txt')
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

    # Add 0
    data.insert(0, 'Ones', 1)

    # set X (training data) and y (target variable)
    # shape (97, 3) (rows, cols)
    cols = data.shape[1]

    # All data from 0 to cols-1
    # training data
    X = data.iloc[:, 0:cols-1]      # training data
    y = data.iloc[:, cols-1:cols]  # target profit

    # convert from data frames to numpy matrices
    # matrix( [[ 17.592  ], [] ....])
    X = np.matrix(X.values)
    y = np.matrix(y.values)

    # theta [[0,0]]
    theta = np.matrix(np.array([0, 0]))

    alpha = 0.01
    iters = 1500

    g, cost = gradientDescent(X, y, theta, alpha, iters)

    print("Found gradient descent by {} and {}".format(g[0, 0], g[0, 1]))
    # print("Error {}".format(computeCost(X, y, g)))

    predict1 = np.dot(g, np.array([1, 3.5])).item(0)  # g[0, 0] + (3.5 * g[0, 1])
    predict2 = np.dot(g, np.array([1, 7])).item(0)

    print("For population = 35'000 we predict a profit of {}".format(predict1 * 10000))
    print("For population = 70'000 we predict a profit of {}".format(predict2 * 10000))
    # Plot
    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    f = g[0, 0] + (g[0, 1] * x)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.Population, data.Profit, label='Traning Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(iters), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()
