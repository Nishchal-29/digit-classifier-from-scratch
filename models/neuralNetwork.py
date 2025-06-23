class NeuralNetwork:
    def __inti__(self, layers):
        self.layers = layers
    
    def forward(self, X):
        A = X.T
        for layer in self.layers:
            A = layer.forward(A)
        return A
    
    def backward(self, Y_hat, Y_true):
        dA = self.layers[-1].backward(Y_hat, Y_true)
        for layer in reversed(self.layers[:-1]):
            dA = layer.backward(dA)
        return dA
    
    def get_params_and_grads(self):
        params = {}
        grads = {}
        cnt = 1
        for layer in self.layers:
            if hasattr(layer, 'params'):
                layer_params = layer.params()
                layer_grads = layer.grads()
                params[f"W{cnt}"] = layer_params['W']
                params[f"b{cnt}"] = layer_params['b']
                grads[f"dW{cnt}"] = layer_grads['dW']
                grads[f"db{cnt}"] = layer_grads['db']
                cnt += 1
        return params, grads  