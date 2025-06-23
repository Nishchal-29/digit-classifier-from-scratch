import numpy as np

def compute_accuracy(logits, labels):
    # logits shape: (num_classes, batch_size)
    # labels shape: (batch_size,)
    preds = np.argmax(logits, axis=0)
    accuracy = np.mean(preds == labels)
    return accuracy