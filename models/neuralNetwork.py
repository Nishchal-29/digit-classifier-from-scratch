class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, X):
        A = X.T
        for layer in self.layers:
            A = layer.forward(A)
        return A
    
    def backward(self, dLoss):
        dA = dLoss
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def get_params_and_grads(self):
        params = {}
        grads = {}
        count = 1
        for layer in self.layers:
            if hasattr(layer, 'params'):
                p = layer.params()
                g = layer.grads()
                params[f"W{count}"] = p["W"]
                params[f"b{count}"] = p["b"]
                grads[f"dW{count}"] = g["dW"]
                grads[f"db{count}"] = g["db"]
                count += 1
        return params, grads
