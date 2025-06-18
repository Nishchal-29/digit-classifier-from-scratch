import numpy as np

class ReLU:
    def __init__(self):
        self.cache_input = None
        
    def forward(self, Z):
        self.cache_input = Z
        return np.maximum(0, Z)
    
    def backward(self, dA):
        Z = self.cache_input
        dZ = dA * (Z > 0)
        return dZ
    
class LeakyReLU:
    def __init__(self, alpha = 0.2):
        self.alpha = alpha
        self.cache_input = None
        
    def forward(self, Z):
        self.cache_input = Z
        return np.where(Z > 0, Z, self.alpha * Z)
    
    def backward(self, dA):
        Z = self.cache_input
        dZ = dA * np.where(Z > 0, 1, self.alpha)
        return dZ
    
class Softmax:
    def __init__(self):
        self.input_cache = None
        
    def forward(self, Z):
        self.input_cache = Z
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)
    
    def backward(self, Y_hat, Y_true):
        Y = np.zeros_like(Y_hat)
        Y[Y_true, np.arange(Y_true.size)] = 1
        return Y_hat - Y
    
    