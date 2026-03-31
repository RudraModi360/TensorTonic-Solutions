import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    First-Visit Monte Carlo Policy Evaluation

    Args:
        episodes: list of episodes, each episode is list of (state, reward)
        gamma: discount factor
        n_states: total number of states

    Returns:
        V: np.array of shape (n_states,)
    """

    # Sum of returns and count per state
    returns_sum = np.zeros(n_states, dtype=np.float64)
    returns_count = np.zeros(n_states, dtype=np.int32)

    for episode in episodes:
        states = [s for (s, r) in episode]
        rewards = [r for (s, r) in episode]

        T = len(episode)
        returns = np.zeros(T, dtype=np.float64)

        # Step 1: compute returns backward
        G = 0
        for t in reversed(range(T)):
            G = rewards[t] + gamma * G
            returns[t] = G

        # Step 2: first-visit tracking
        visited = set()

        for t in range(T):
            s = states[t]

            if s not in visited:
                visited.add(s)

                returns_sum[s] += returns[t]
                returns_count[s] += 1

    # Step 3: compute value function
    V = np.zeros(n_states, dtype=np.float64)

    for s in range(n_states):
        if returns_count[s] > 0:
            V[s] = returns_sum[s] / returns_count[s]

    return V