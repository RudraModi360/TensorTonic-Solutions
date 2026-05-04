def matrix_factorization_sgd_step(U, V, r, lr, reg):
    """
    Perform one SGD step for matrix factorization.

    U: list of floats (user latent vector)
    V: list of floats (item latent vector)
    r: actual rating
    lr: learning rate
    reg: regularization strength

    Returns:
        (U_new, V_new)
    """

    # Step 1: compute dot product
    pred = sum(u * v for u, v in zip(U, V))

    # Step 2: compute error
    error = r - pred

    # Step 3: copy original values (IMPORTANT)
    U_old = U[:]
    V_old = V[:]

    # Step 4: update simultaneously
    U_new = []
    V_new = []

    for i in range(len(U)):
        u_i = U_old[i]
        v_i = V_old[i]

        new_u = u_i + lr * (error * v_i - reg * u_i)
        new_v = v_i + lr * (error * u_i - reg * v_i)

        U_new.append(new_u)
        V_new.append(new_v)

    return U_new, V_new