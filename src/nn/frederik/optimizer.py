class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, params):
        for p in params:
            p.data -= self.learning_rate * p.grad
