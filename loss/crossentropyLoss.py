import numpy as np

def cross_entropy(logits, labels):
    shifted_logits = logits - np.max(logits, axis=0, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)

    N = labels.shape[0]
    one_hot = np.zeros_like(probs.T)
    one_hot[np.arange(N), labels] = 1
    one_hot = one_hot.T

    log_likelihood = -np.log(probs[labels, np.arange(N)] + 1e-9)
    loss = np.sum(log_likelihood) / N
    dlogits = (probs - one_hot) / N

    return loss, dlogits
