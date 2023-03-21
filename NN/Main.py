import numpy as np

from LossFunctions import cost_function
from NetworkFunctions import init_params, update_parameters
from Propogation import forward_prop, backprop


def train(X, Y, layer_dims, epochs, lr):
    params = init_params(layer_dims)
    cost_history = []

    for i in range(epochs):
        Y_hat, caches = forward_prop(X, params)
        cost = cost_function(Y_hat, Y)
        cost_history.append(cost)
        grads = backprop(Y_hat, Y, caches)

        params = update_parameters(params, grads, lr)

    return params, cost_history