import numpy as np

def init_params(layer_dims):
    np.random.seed(3)
    params = {}
    L = len(layer_dims)

    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return params

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads['W' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['b' + str(l + 1)]

    return parameters

