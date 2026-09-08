import numpy as np


class SkeletonNN:

    def __init__(self , input_dim= 784 , hidden_dim = 4 , output_dim = 10, seed = 0):
        rng = np.random.default_rng(seed)

        self.W = rng.standard_normal((input_dim , output_dim)) * 0.01
        self.b = np.zeros(output_dim)

        self.x = None
        self.Y = None

        self.grad_W = None
        self.grad_b = None


    def forward(self, x):
        self.x = x
        return (self.x @ self.W) + self.b

    def backward(self , grad_output):
        self.grad_W = self.x.T @ grad_output
        self.grad_b = grad_output.sum(axis=0)
        return grad_output @ self.W.T

    def params(self):
        return [ (self.W, self.grad_W) , (self.b , self.grad_b)]

