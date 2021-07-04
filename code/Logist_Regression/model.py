import numpy as np

from initialize_with_zeros import initialize_with_zeros
from optimize import optimize
from predict import predict


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    dim = X_train.shape[0]

    w, b = initialize_with_zeros(dim)

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations=num_iterations,
                                        learning_rate=learning_rate, print_cost=print_cost)

    w = parameters['w']
    b = parameters['b']

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print(f"train accyracy {100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100}")
    print(f"test accuracy {100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100}")

    d = dict(
        costs=costs,
        Y_prediction_test=Y_prediction_test,
        Y_prediction_train=Y_prediction_train,
        w=w,
        b=b,
        learning_rate=learning_rate,
        num_iterations=num_iterations
    )

    return d
