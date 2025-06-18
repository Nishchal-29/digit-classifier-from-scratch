import numpy as np

class Adam:
    def __init__(self, parameters, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.parameters = parameters
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.t = 0
        
        self.m = {key: np.zeros_like(value) for key, value in parameters.items()}
        self.v = {key: np.zeros_like(value) for key, value in parameters.items()}
        
        
    def step(self, grads):
        self.t += 1
        for param_key in self.parameters:
            grad_key = "d" + param_key
            if grad_key in grads:
                g = grads[grad_key]
                
                self.m[param_key] = self.beta1 * self.m[param_key] + (1 - self.beta1) * g
                self.v[param_key] = self.beta2 * self.v[param_key] + (1 - self.beta2) * (g ** 2)
                
                m_hat = self.m[param_key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[param_key] / (1 - self.beta2 ** self.t)
                
                self.parameters[param_key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)