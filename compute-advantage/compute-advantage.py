import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    states = np.array(states)
    rewards = np.array(rewards)
    V = np.array(V)

    n = len(rewards)
    returns = np.zeros(n)

    # Step 1: compute returns backward
    running_return = 0
    for t in reversed(range(n)):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    # Step 2: compute advantage
    advantages = returns - V[states]

    return advantages