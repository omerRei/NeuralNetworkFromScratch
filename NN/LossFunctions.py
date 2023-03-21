import numpy as np


def cost_function(A, Y):
    m = Y.shape[1]

    cost = (-1 / m) * (np.dot(np.log(A), Y.T) + np.dot(np.log(1 - A), 1 - Y.T))

    return cost