import numpy as np

from sigmoid import sigmoid


def predict(w, b, X):
    m = X.shape[1]

    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):

        if A[:, i] <= 0.5:
            Y_prediction[:, i] = 0
        else:
            Y_prediction[:, i] = 1

    assert (Y_prediction.shape == (1, m))

    return Y_prediction
