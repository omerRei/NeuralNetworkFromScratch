import numpy as np

# Z (linear hypothesis) - Z = W*X + b ,
# W - weight matrix, b- bias vector, X- Input

def sigmoid(Z):
    A = 1 / (1 + np.exp(np.dot(-1, Z)))
    cache = (Z)
    return A, cache