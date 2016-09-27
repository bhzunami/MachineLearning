#!/usr/bin/env python
"""
AUTHOR
    N. Mauchle <nicolas@nicolasmauchle.ch>

LICENSE
    MIT

VERSION
    1.0

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from ex1 import computeCost, gradientDescent

if __name__ == "__main__":
    path = os.path.join(os.getcwd(), 'data/ex1data2.txt')
    data = pd.read_csv(path, header=None, names=['Sizes', 'Bedrooms', 'Price'])

    # Feature normilize
    data = (data - data.mean()) / data.std()

    data.insert(0, 'Ones', 1)

    # set training data
    cols = data.shape[1]

    X = data.iloc[:, 0:cols-1]
    y = data.iloc[:, cols-1:cols]

    X = np.matrix(X.values)
    y = np.matrix(y.values)

    theta = np.matrix(np.array([0, 0, 0]))
    iters = 1500
    alpha = 0.01

    g, cost = gradientDescent(X, y, theta, alpha, iters)
    print(g)
    # cost (error) function
    print(computeCost(X, y, g))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(iters), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')

    plt.show()
