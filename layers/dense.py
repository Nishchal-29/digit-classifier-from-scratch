import numpy as np

class Linear:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(out_dim, in_dim) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros((out_dim, 1))
        self.dW = None
        self.db = None
        self.cache_input = None
        
    def forward(self, X):
        self.cache_input = X
        return np.dot(self.W, X) + self.b
    
    def backward(self, dZ):
        A_prev = self.cache_input
        m = A_prev.shape[1]
        self.dW = np.dot(dZ, A_prev.T) / m
        self.db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(self.W.T, dZ)
        return dA_prev
    
    def params(self):
        return {'W': self.W, 'b': self.b}
    
    def grads(self):
        return {'dW': self.dW, 'db': self.db}