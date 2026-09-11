from .base import Component


class Sequential(Component):
    def __init__(self, components) -> None:
        self.components = list(components)

    def forward(self, x):
        for component in self.components:
            x = component.forward(x)
        return x

    def backward(self, d_y):
        for component in reversed(self.components):
            d_y = component.backward(d_y)
        return d_y

    def params(self):
        return [
            parameter
            for component in self.components
            for parameter in component.params()
        ]
