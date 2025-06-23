import numpy as np
import matplotlib.pyplot as plt
import random
import time
from sklearn.model_selection import train_test_split
from torchvision import transforms, datasets

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root = './data', train = True, transform = transform, download = True)
test_dataset = datasets.MNIST(root = './data', train = False, transform = transform, download = True)

train_images = train_dataset.data.numpy().reshape(-1, 28*28) / 255.0
train_labels = train_dataset.targets.numpy()

test_images = test_dataset.data.numpy().reshape(-1, 28*28) / 255.0
test_labels = test_dataset.targets.numpy()

X_train, X_val, Y_train, Y_val = train_test_split(train_images, train_labels, test_size = 0.2, random_state = 42)

# importing the custom modules
from layers.dense import Linear
from layers.activations import ReLU, LeakyReLU, Softmax
from layers.norm import BatchNorm
from utils.metrics import compute_accuracy
from utils.visualize import show_examples
from optimizers.AdamOptimizer import Adam
from optimizers.SGDOptimizer import SGD
from optimizers.RMSpropOptimizer import RMSprop
from loss.crossentropyLoss import cross_entropy
from models.neuralNetwork import NeuralNetwork

def train(model, optimizer, X_train, y_train, X_val, y_val, epochs, batch_size):
    num_samples = X_train.shape[0]
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        start_time = time.time()
        permutation = np.random.permutation(num_samples)
        X_train = X_train[permutation]
        y_train = y_train[permutation]
        
        epoch_loss = 0
        epoch_accuracy = 0
        
        for i in range(0, num_samples, batch_size):
            X_batch = X_train[i : i+batch_size]
            y_batch = y_train[i : i+batch_size]
            if X_batch.shape[0] == 0:
                continue
            
            # Forward pass
            logits = model.forward(X_batch)
            loss, dlogits = cross_entropy(logits, y_batch)
            accuracy = compute_accuracy(logits, y_batch)
            
            # Backward pass
            model.backward(dlogits)
            _, grads = model.get_params_and_grads()
            optimizer.step(grads)
            
            epoch_loss += loss * X_batch.shape[0]
            epoch_accuracy += accuracy * X_batch.shape[0]
            
        avg_loss = epoch_loss / num_samples
        avg_accuracy = epoch_accuracy / num_samples
        
        # Evaluation on validation set
        val_logits = model.forward(X_val)
        val_loss, _ = cross_entropy(val_logits, Y_val)
        val_accuracy = compute_accuracy(val_logits, Y_val)

        train_losses.append(avg_loss)
        train_accuracies.append(avg_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        end_time = time.time()
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Time Taken: {end_time - start_time:.2f}s")
        
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.xlabel("Epoch")
    plt.legend()
    plt.title("Training Progress")
    plt.show()

layers = [
    Linear(784, 64),
    LeakyReLU(),
    Linear(64, 32),
    BatchNorm(32),
    LeakyReLU(),
    Linear(32, 10),
]

model = NeuralNetwork(layers)

params, _ = model.get_params_and_grads()
optimizer = RMSprop(params, learning_rate=0.01)
train(model, optimizer, X_train, Y_train, X_val, Y_val, epochs=10, batch_size=64)

def test_model(model, X_test, y_test):
    logits = model.forward(X_test)
    y_pred = np.argmax(logits, axis = 0)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy:.4f} or {accuracy * 100:.2f}%")
    return y_pred

y_test_pred = test_model(model, test_images, test_labels)
show_examples(test_images, test_labels, y_test_pred, True, 10)
show_examples(test_images, test_labels, y_test_pred, False, 10)
