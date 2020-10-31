import numpy as np


class Optim(object):
    """Optimizers"""

    def __init__(self, learning_rate=0.01, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta

    def adjust_params(self, model):
        raise NotImplementedError

class GradientDescent(Optim):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate=learning_rate)

    def adjust_params(self, model):
        from .nn import Layer
        layers = model.layers

        for layer in layers:
            if isinstance(layer, Layer):
                layer.weights = layer.weights - layer.weights_delta * self.learning_rate
                layer.bias = layer.bias - layer.bias_delta * self.learning_rate



class Momentum(Optim):
    def __init__(self, learning_rate=0.01, beta=0.9):
        super().__init__(learning_rate=learning_rate, beta=beta)

    def adjust_params(self, model):
        from .nn import Layer
        layers = model.layers

        for layer in layers:
            if isinstance(layer, Layer):
                if "momentum_weights_delta" not in layer.extras:
                    layer.extras["momentum_weights_delta"] = 0
                if "momentum_bias_delta" not in layer.extras:
                    layer.extras["momentum_bias_delta"] = 0

                momentum_weights_delta = layer.extras["momentum_weights_delta"]
                momentum_bias_delta = layer.extras["momentum_bias_delta"]

                layer.extras["momentum_weights_delta"] = self.beta * momentum_weights_delta + (1 - self.beta) * layer.weights_delta.value
                layer.extras["momentum_bias_delta"] = self.beta * momentum_bias_delta  + (1 - self.beta) * layer.bias_delta.value

                layer.weights = layer.weights - layer.extras["momentum_weights_delta"] * self.learning_rate
                layer.bias = layer.bias - layer.extras["momentum_bias_delta"] * self.learning_rate

class RMSProp(Optim):
    def __init__(self, learning_rate=0.01, beta=0.9):
        super().__init__(learning_rate=learning_rate, beta=beta)

    def adjust_params(self, model):
        from .nn import Layer
        layers = model.layers

        for layer in layers:
            if isinstance(layer, Layer):
                if "rms_weights_delta" not in layer.extras:
                    layer.extras["rms_weights_delta"] = 0
                if "rms_bias_delta" not in layer.extras:
                    layer.extras["rms_bias_delta"] = 0

                rms_weights_delta = layer.extras["rms_weights_delta"]
                rms_bias_delta = layer.extras["rms_bias_delta"]

                layer.extras["rms_weights_delta"] = self.beta * rms_weights_delta + (1 - self.beta) * layer.weights_delta.value ** 2
                layer.extras["rms_bias_delta"] = self.beta * rms_bias_delta  + (1 - self.beta) * layer.bias_delta.value ** 2

                layer.weights = layer.weights - (layer.weights_delta / np.sqrt(layer.extras["rms_weights_delta"])) * self.learning_rate
                layer.bias = layer.bias - (layer.bias_delta / np.sqrt(layer.extras["rms_bias_delta"])) * self.learning_rate


class Adam(Optim):
    def __init__(self, learning_rate=0.01, beta=0.9):
        super().__init__(learning_rate=learning_rate, beta=beta)