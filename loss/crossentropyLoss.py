import numpy as np

def cross_entropy_loss(Y_hat, Y_true):
    m = Y_true.shape[0]
    log_likelihood = -np.log(Y_hat[Y_true, np.arrange(m)] + 1e-15)
    loss = np.sum(log_likelihood) / m
    return loss