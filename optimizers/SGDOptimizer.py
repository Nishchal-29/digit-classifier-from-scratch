import numpy as np

class SGD:
    def __init__(self, parameters, learning_rate = 0.01, jitter = 1e-4):
        self.parameters = parameters
        self.lr = learning_rate
        self.jitter = jitter
        
    def step(self, grads):
        for param_key in self.parameters:
            grad_key = "d" + param_key
            if grad_key in grads:
                self.parameters[param_key][:] -= (self.lr + self.jitter * np.random.randn(1, 1)) * grads[grad_key]