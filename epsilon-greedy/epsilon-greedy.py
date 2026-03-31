import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    """
    ε-greedy action selection

    Args:
        q_values: 1D array of Q-values
        epsilon: exploration probability
        rng: optional np.random.Generator

    Returns:
        action: int
    """

    if rng is None:
        rng = np.random

    n_actions = len(q_values)

    # Step 1: decide explore vs exploit
    if rng.random() < epsilon:
        # Explore: random action
        action = rng.integers(n_actions) if hasattr(rng, "integers") else rng.randint(n_actions)
    else:
        # Exploit: greedy action
        action = int(np.argmax(q_values))

    return action