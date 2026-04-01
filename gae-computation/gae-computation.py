def gae(rewards, values, gamma, lam):
    """
    Compute Generalized Advantage Estimation.
    """
    T = len(rewards)
    advantages = [0] * T
    delta = [0] * T

    # Step 1: compute all deltas
    for t in range(T):
        td_target = rewards[t] + gamma * values[t + 1]
        delta[t] = td_target - values[t]

    # Step 2: backward recursion
    advantages[T - 1] = delta[T - 1]
    for t in reversed(range(T - 1)):
        advantages[t] = delta[t] + gamma * lam * advantages[t + 1]

    return advantages