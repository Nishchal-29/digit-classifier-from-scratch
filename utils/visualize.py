import matplotlib.pyplot as plt
import numpy as np

def show_examples(X, Y_true, Y_hat, num = 10):
    fig, axes = plt.subplots(1, num, figsize=(15, 2))
    for i in range(num):
        img = X[:, i].reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"T:{Y_true[i]}\nP:{Y_hat[i]}")
    plt.tight_layout()
    plt.show()