from propagate import propagate


def optimize(w, b, X, Y, num_iterations=10, learning_rate=1e4, print_cost=False):
    costs = list()

    dw = None
    db = None

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads['dw']
        db = grads['db']

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i} :  {cost}")

    params = dict(
        w=w,
        b=b
    )
    grads = dict(
        dw=dw,
        db=db
    )

    return params, grads, costs
