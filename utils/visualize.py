import matplotlib.pyplot as plt
import random
import numpy as np

def show_examples(X, Y_true, Y_pred, correct=True, n=5):
    indices = [i for i in range(len(Y_true)) if (Y_pred[i] == Y_true[i]) == correct]
    selected = random.sample(indices, min(n, len(indices)))

    plt.figure(figsize=(10, 4))
    for i, idx in enumerate(selected):
        plt.subplot(1, n, i+1)
        plt.imshow(X[idx].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title(f"Label: {Y_true[idx]}\nPred: {Y_pred[idx]}")
    plt.suptitle("Correctly Classified" if correct else "Misclassified", fontsize=14)
    plt.show()