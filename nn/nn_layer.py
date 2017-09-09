from __future__ import print_function
from __future__ import division
import numpy as np
import math


def myrandom_array2d(d1, d2):
    return np.random.randn(d1, d2)


def myrandom_vector(d1):
    return np.random.randn(d1)


def calc_softmax(z):
    tmp = np.exp(z)
    total = sum(tmp)
    return tmp/total


class Layer(object):
    def __init__(self, name, size):
        self.name = name
        self.size = size
        # the output of current layer, usually is a vector of the activated result
        self.output = None
        self.input_layer = None
        self.next_layer = None
        return

    def init(self):
        """init the weight matrix"""
        pass

    def set_input_layer(self, input_layer):
        self.input_layer = input_layer
        return

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer
        return

    def get_output(self):
        return self.output

    def get_size(self):
        return self.size

    def __str__(self):
        return "%d\t%s" % (self.size, self.name)

    def detail_info(self):
        fan_in = 0
        if self.input_layer is not None:
            fan_in = self.input_layer.get_size()

        msg = "[%d, %d] %s" % (fan_in, self.size, self.name)
        return msg


class InputLayer(Layer):
    def __init__(self, name, size):
        super(InputLayer, self).__init__(name, size)
        return

    def init(self):
        """do nothing"""
        # self.output = np.zeros(self.size)
        return

    def feed(self, data):
        self.output = data
        return


class ActiveLayer(Layer):
    def __init__(self, name, size):
        super(ActiveLayer, self).__init__(name, size)
        self.weights = None
        self.bias = None
        self.z = None
        self.sigma = None
        self.delta_weights = None
        return

    def init(self):
        fan_in = self.input_layer.get_size()
        self.weights = myrandom_array2d(fan_in, self.size)
        self.bias = myrandom_vector(self.size)

        # forward results
        self.z = np.zeros(self.size)
        self.output = np.zeros(self.size)

        # backward results
        self.sigma = np.zeros(self.size)
        self.delta_weights = np.zeros((fan_in, self.size))
        return

    def active(self):
        pass

    def calc_error(self):
        pass

    def get_sigma(self):
        return self.sigma

    def get_weights(self):
        return self.weights

    def update_weights(self, lr):
        self.weights -= lr * self.delta_weights
        self.bias -= lr * self.sigma
        return


class SoftmaxOutputLayer(ActiveLayer):
    def __init__(self, name, size):
        super(SoftmaxOutputLayer, self).__init__(name, size)
        return

    def init(self):
        """do nothing"""
        return

    def active(self):
        x = self.input_layer.get_output()
        self.output = calc_softmax(x)
        return

    def calc_error(self, labels):
        self.sigma = self.output - labels
        return

    def calc_cost(self, labels):
        i = np.argmax(labels)
        y = self.output[i]
        if y < 0.00000001:
            return 10000000
        elif y > 0.999:
            return 0

        return -1*math.log(y)

    def update_weights(self, lr):
        """do nothing"""
        return


class HiddenLayer(ActiveLayer):
    def __init__(self, name, size, activefunc):
        super(HiddenLayer, self).__init__(name, size)
        self.func = activefunc
        return

    def active(self):
        x = self.input_layer.get_output()
        np.dot(x, self.weights, out=self.z)
        self.z += self.bias

        self.output = self.func.forward(self.z)
        #if check_toobig(self.output):
        #    print("%s too big" % (self.name,))
        return

    def calc_error(self):
        # 1. calc sigma
        next_sigma = self.next_layer.get_sigma()
        next_weights = self.next_layer.get_weights()

        if next_weights is None:
            self.sigma = np.copy(next_sigma)
        else:
            #print("weight.shape=%s, next=%s, my=%s" % (next_weights.shape, next_sigma.shape, self.sigma.shape))
            np.dot(next_weights, next_sigma, out=self.sigma)
        self.sigma = self.sigma * self.func.backward(self.output)

        # 2. calc delta_weights
        x = self.input_layer.get_output()
        self.delta_weights = np.dot(x.reshape((-1, 1)), self.sigma.reshape((1, -1)))
        #print("name=%s, delta_weight=%s" % (self.name, self.delta_weights.shape))
        return


def check_toobig(y):
    for i in range(y.shape[0]):
        z = y[i]
        if z*z > 10000:
            print("z=%s" % (z, ))
            return True
    return False
