def _dot(a, b):
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def _vec_add(a, b):
    return [x + y for x, y in zip(a, b)]


def _vec_sub(a, b):
    return [x - y for x, y in zip(a, b)]


def _scalar_mul(s, v):
    return [s * x for x in v]


def lbfgs_direction(grad, s_list, y_list):
    """
    Compute the L-BFGS search direction using the two-loop recursion.
    """
    m = len(s_list)
    q = grad[:]   # copy
    alpha = [0] * m
    rho = [0] * m

    # ---- First Loop (backward) ----
    for i in range(m - 1, -1, -1):
        rho[i] = 1.0 / _dot(y_list[i], s_list[i])
        alpha[i] = rho[i] * _dot(s_list[i], q)
        q = _vec_sub(q, _scalar_mul(alpha[i], y_list[i]))

    # ---- Initial scaling ----
    if m > 0:
        s = s_list[-1]
        y = y_list[-1]
        gamma = _dot(s, y) / _dot(y, y)
    else:
        gamma = 1.0

    r = _scalar_mul(gamma, q)

    # ---- Second Loop (forward) ----
    for i in range(m):
        beta = rho[i] * _dot(y_list[i], r)
        r = _vec_add(r, _scalar_mul(alpha[i] - beta, s_list[i]))

    # ---- Descent direction ----
    direction = _scalar_mul(-1, r)

    return direction