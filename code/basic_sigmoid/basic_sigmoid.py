import math

import numpy as np


def basic_sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def sigmoid_derivative(x):
    ds = sigmoid(x) * (1 - sigmoid(x))
    return ds


value =  255
print(sigmoid(value))
