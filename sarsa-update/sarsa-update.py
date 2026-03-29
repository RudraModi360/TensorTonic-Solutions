def sarsa_update(q_table, state, action, reward, next_state, next_action, alpha, gamma):
    
    td_target = reward + gamma * q_table[next_state][next_action]
    td_error = td_target - q_table[state][action]

    q_table[state][action] += alpha * td_error

    return q_table