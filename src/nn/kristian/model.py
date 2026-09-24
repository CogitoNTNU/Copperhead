from src.nn.kristian.module import Module


class Sequential(Module):
    def __init__(self, *modules: Module):
        self.modules = list(modules)

    def forward(self, x):
        out = x

        for m in self.modules:
            out = m.forward(out)

        return out

    def backward(self, dout):
        out = dout

        for m in reversed(self.modules):
            out = m.backward(out)

        return out

    def parameters(self):
        return [p for m in self.modules for p in m.parameters()]
