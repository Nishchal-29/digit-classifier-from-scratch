import numpy as np

class RMSprop:
    def __init__(self, parameters, learning_rate = 0.001, beta = 0.9, epsilon = 1e-8):
        self.parameters = parameters
        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.s = {key: np.zeros_like(value) for key, value in parameters.items()}
        
    def step(self, grads):
        for param_key in self.parameters:
            grad_key = "d" + param_key
            if grad_key in grads:
                g = grads[grad_key]
                
            self.s[param_key] = self.beta * self.s[param_key] + (1 - self.beta) * (g ** 2)
            self.parameters[param_key] -= self.lr * g / (np.sqrt(self.s[param_key]) + self.epsilon)