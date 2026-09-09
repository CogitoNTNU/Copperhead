class KristianSgd:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def step(self, params):
        for parameter, gradient in params:
            parameter -= self.learning_rate * gradient
