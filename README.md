# Digit Classifier – Neural Network from Scratch

This project demonstrates a complete digit classification system built **entirely from scratch**, without using high-level machine learning libraries like TensorFlow or PyTorch for model training. The model is trained on the MNIST dataset and achieves **98.26% training accuracy** and **97.2% testing accuracy**.

---

## Project Overview

The goal of this project is to understand the inner workings of neural networks by implementing every component from the ground up. This includes:

- Custom dense layers
- Activation functions (ReLU, LeakyReLU, Softmax)
- Batch normalization
- Multiple optimizers (SGD, RMSprop, Adam)
- Cross-entropy loss
- Forward and backward propagation
- Accuracy tracking and prediction visualization

All components are modularized into separate files to ensure clean architecture and reusability.

---

## Model Architecture

The model is a multi-layer perceptron with the following architecture:

```
Input (784) → Linear(64) → BatchNorm → ReLU
           → Linear(32) → ReLU
           → Linear(10) → Softmax
```

---

## Project Structure

```
digit-classifier-from-scratch/
│
├── main.py                     # Main script for training and evaluation
│
├── layers/
│   ├── dense.py                # Linear (fully connected) layer
│   ├── activation.py           # ReLU, LeakyReLU, Softmax
│   └── norm.py                 # BatchNorm
│
├── loss/
│   └── loss.py                 # Cross-entropy loss and gradient
│
├── optim/
│   ├── sgd.py                  # Gradient Descent and Momentum
│   ├── adam.py                 # Adam optimizer
│   └── rmsprop.py              # RMSprop optimizer
│
├── models/
│   └── neural_net.py           # Neural network architecture and training logic
│
└── utils/
    ├── metrics.py              # Accuracy computation
    └── visualize.py            # Prediction visualization
```

---

## Key Highlights

- Every function and class was written from scratch using **NumPy**.
- The `NeuralNetwork` class dynamically chains all layers and performs full forward and backward propagation.
- Optimizers are plug-and-play, requiring only parameters and gradients.
- The structure is designed to be extensible and readable, making it easy to experiment with new layers or training strategies.

---

## Results

- **Training Accuracy**: 98.26%
- **Testing Accuracy**: 97.2%
- Dataset: [MNIST handwritten digits](http://yann.lecun.com/exdb/mnist/)

---

### To Train and Evaluate:

```bash
python main.py
```

This will automatically load the dataset, train the model, and display predictions visually at the end.

---
