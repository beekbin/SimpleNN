
from __future__ import division
from __future__ import print_function
import numpy as np


class ActiveFunction(object):
    def __init__(self, name):
        self.name = name

    def forward(self, x):
        pass

    def backward(self, y):
        pass

    def __str__(self):
        return self.name


class SigmoidActiveFunction(ActiveFunction):
    def __init__(self):
        super(self.__class__, self).__init__("sigmoid")
        return

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, y):
        return y * (1-y)


class TanhActiveFunction(ActiveFunction):
    def __init__(self):
        super(self.__class__, self).__init__("tanh")
        return

    def forward(self, x):
        return np.tanh(x)

    def backward(self, y):
        return 1 - y*y


class ReluActiveFunction(ActiveFunction):
    def __init__(self):
        super(self.__class__, self).__init__("relu")
        return

    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, y):
        return 1


class LinearActiveFunction(ActiveFunction):
    def __init__(self):
        super(self.__class__, self).__init__("linear")
        return

    def forward(self, x):
        return x

    def backward(self, y):
        return 1


tanhFunc = TanhActiveFunction()
sigmoidFunc = SigmoidActiveFunction()
reluFunc = ReluActiveFunction()
linearFunc = LinearActiveFunction()