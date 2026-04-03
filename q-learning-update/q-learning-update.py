import numpy as np

def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    Returns: updated Q-table Q_new
    """
    Q=np.array(Q,dtype=float)
    best_next = np.max(Q[s_next])   # O(n_actions)
    
    td_target = r + gamma * best_next
    td_error = td_target - Q[s, a]
    
    Q[s, a] += alpha * td_error
    
    return Q