def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    
    for step in range(steps):
        grad = 2 * a * x0 + b
        x0 = x0 - lr * grad
        
    return x0