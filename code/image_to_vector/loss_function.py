import numpy as np


def L1_loss_funcion(y_hat, y):
    s = np.sum(np.abs(y - y_hat))
    return s


def L2_loss_funcion(y_hat, y):
    s = np.dot(y - y_hat, y - y_hat)

    return s
