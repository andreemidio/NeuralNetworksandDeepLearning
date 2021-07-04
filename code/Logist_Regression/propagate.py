import numpy as np

from sigmoid import sigmoid


def propagate(w, b, X, Y):
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dw = 1 / m * np.dot(X, (A - Y).T)

    db = 1 / m * np.sum(A - Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)

    cost = np.squeeze(cost)

    assert (cost.shape == ())

    grads = dict(
        dw=dw,
        db=db
    )

    return grads, cost
