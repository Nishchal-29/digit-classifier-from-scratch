import numpy as np

class BatchNorm:
    def __init__(self, dim, momentum=0.9, epsilon=1e-5):
        self.dim = dim
        self.momentum = momentum
        self.epsilon = epsilon
        
        self.gamma = np.ones((dim, 1))
        self.beta = np.zeros((dim, 1))
        
        self.running_mean = np.zeros((dim, 1))
        self.running_var = np.ones((dim, 1))
        
        self.training = True
        self.cache_input = None
        
        self.dgamma = np.zeros((dim, 1))
        self.dbeta = np.zeros((dim, 1))
        
    def forward(self, X):
        self.cache_input = X
        if self.training:
            self.batch_mean = np.mean(X, axis=1, keepdims=True)
            self.batch_var = np.var(X, axis=1, keepdims=True)
            
            self.normalized_X = (X - self.batch_mean)/(np.sqrt(self.batch_var + self.epsilon))
            out = self.gamma * self.normalized_X + self.beta
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
            
        else:
            normalized_X = (X - self.running_mean) / (np.sqrt(self.running_var + self.epsilon))
            out = self.gamma * normalized_X + self.beta
            
        return out
    
    def backward(self, d_out):
        N = d_out.shape[1]
        d_normalized_X = d_out * self.gamma
        d_var = np.sum(d_normalized_X * (self.cache_input - self.batch_mean) * -0.5 * np.power(self.batch_var + self.epsilon, -1.5), axis=1, keepdims=True)
        d_mean = np.sum(d_normalized_X * -1 / np.sqrt(self.batch_var + self.epsilon), axis=1, keepdims=True) + d_var * np.mean(-2 * (self.cache_input - self.batch_mean), axis=1, keepdims=True)
        
        d_input = (d_normalized_X / (np.sqrt(self.batch_var + self.epsilon))) + (d_var * 2 * (self.cache_input - self.batch_mean) / N) + (d_mean / N)
        self.dgamma = np.sum(d_out * self.normalized_X, axis=1, keepdims=True)
        self.dbeta = np.sum(d_out, axis=1, keepdims=True)
        
        return d_input
    
    def params(self):
        return {"W": self.gamma, "b": self.beta}

    def grads(self):
        return {"dW": self.dgamma, "db": self.dbeta}
    
    def train(self):
        self.training = True
        
    def eval(self):
        self.training = False
