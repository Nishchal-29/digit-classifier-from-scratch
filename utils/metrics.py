import numpy as np

def compute_accuracy(Y_hat, Y_true):
    pred_labels = np.argmax(Y_hat, axis=0)
    true_labels = np.argmax(Y_true, axis=0)
    accuracy = np.mean(pred_labels == true_labels)
    return accuracy