import numpy as np
from .optim import GradientDescent
from .loss import MeanSquaredError
from .tensor import *

class Model(object):
    def __init__(self, layers=[]):
        self.layers = layers

    def predict(self, x):

        for layer in self.layers:
            x = layer.forward(x)

        return x


    def train(self, inputs, outputs, epochs=100, optim=GradientDescent(), loss=MeanSquaredError()):
        for epoch in range(epochs):
            prediction = self.predict(inputs)
            error = prediction - outputs
            epoch_loss = loss.get_loss(error)
            p = prediction.parents.copy()
            print (f"EPOCH {epoch+1} - LOSS {np.mean(epoch_loss.value)}")

            for layer in self.layers:
                if isinstance(layer, Layer):
                    weights_delta = epoch_loss.gradient(layer.weights)
                    bias_delta = epoch_loss.gradient(layer.bias)
                    layer.weights_delta = weights_delta
                    layer.bias_delta = bias_delta

            optim.adjust_params(self)
            
            
            

    def push(self, layer):
        self.layers.append(layer)

class Layer(object):
    """Layer class"""

    def __init__(self, units, input_shape=[]):
        self.weights = Tensor.randn(input_shape[-1], units)
        self.bias = Tensor.randn(units)
        self.weights_delta =  None
        self.bias_delta = None
        self.input = None
        self.extras = {}

    def forward(self, x):
        raise NotImplementedError

class Activation(object):
    def __init__(self):
        self.input = None

    def forward(self, x):
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, units, input_shape=[]):
        super().__init__(units, input_shape=input_shape)

    def forward(self, x):
        self.input = x
        self.output = x @ self.weights + self.bias
        return self.output

class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.input = x
        return sigmoid(x)

class Softmax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.input = x
        return softmax(x)

class ReLu(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.input = x
        return relu(x)

def sigmoid(t1):
    t = Tensor(1 / (1 + np.exp(-t1.value)))
    t.parents = [t1]

    def sigmoid_backward(v, parents, grad):
        forward_result = 1 / (1 + np.exp(-v.value))
        derivative = forward_result * (1 - forward_result)

        grad.value = grad.value * derivative
        return grad

    t.backward = sigmoid_backward
    return t


def softmax(t1):
    t = Tensor(np.exp(t1.value- np.max(t1.value) / np.sum(np.exp(t1.value- np.max(t1.value)))))
    t.parents = [t1]

    def softmax_backward(v, parents, grad):
        forward_result = np.exp(v.value - np.max(v.value)) / np.sum(np.exp(v.value - np.max(v.value)))
        derivative = forward_result * (1 - forward_result)

        grad.value = grad.value * derivative
        return grad

    t.backward = softmax_backward
    return t


def relu(t1):
    t = Tensor(np.maximum(0, t1.value))
    t.parents = [t1]

    def relu_backward(v,parents,grad):
       forward_result = np.maximum(0, t1.value) 
       derivative = (forward_result > 0).astype(float)

       grad.value = grad.value * derivative
       return grad

    t.backward = relu_backward

    return t