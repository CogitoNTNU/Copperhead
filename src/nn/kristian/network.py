import numpy as np

# Neural network 
class NeuralNetwork:
    def __init__(self, input_dim=784, hidden_dim=2, output_dim=10, seed=0):
        rng = np.random.default_rng(seed)

        activation_functions = ["relu", "softmax"]
        total_dims = [input_dim] + [hidden_dim]

        self.Ws = []
        self.bs = []

        self.grad_Ws = []
        self.grad_bs = []

        # make addition of weights and biases per layer dynamic 
        for x_in, x_out in zip(total_dims[:-1], total_dims[1:]):
            self.Ws.append(rng.standard_normal((x_in, x_out)) * 0.01)
            self.bs.append(np.zeroes(x_out))

        self.x = None
        self.Y = None

    def forward(self, x):
        self.x = x

        # Hidden layer -> ReLu
        
        self.z1 = (self.x @ self.W1) + self.b1

        self.z2 = (self.a1 @ self.W2) + self.b2

        return self.softmax(self.z2)

    # Todo: Update backprop to have a hidden layer 
    def backward(self, grad_output):
        self.grad_W1 = self.x.T @ grad_output
        self.grad_b1 = grad_output.sum(axis=0)
        return grad_output @ self.W1.T

    def params(self):
        return [(self.W1, self.grad_W1), (self.b1, self.grad_b1)]
 
    # Activation functions  
    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis = 1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

    def reLu(self, z):
        return np.maximum(0, z)

