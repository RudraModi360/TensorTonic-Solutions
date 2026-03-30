import numpy as np

def policy_gradient_loss(log_probs, rewards, gamma):
    log_probs = np.array(log_probs)
    rewards = np.array(rewards)

    # Reward-to-go
    returns = np.zeros_like(rewards, dtype=np.float64)
    G = 0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G

    # Baseline
    baseline = np.mean(returns)

    # Advantage
    advantages = returns - baseline

    # ✅ Use MEAN (this fixes your issue)
    loss = -np.mean(log_probs * advantages)

    return loss.tolist()