from __future__ import division
from __future__ import print_function
import numpy as np

import nn_layer
import activation


class NNetwork(object):
    def __init__(self):
        self.input_layer = None
        self.output_layer = None
        self.hidden_layers = []
        return

    def set_input(self, input_layer):
        self.input_layer = input_layer
        return

    def add_hidden_layer(self, hid_layer):
        self.hidden_layers.append(hid_layer)
        return

    def set_output(self, output_layer):
        self.output_layer = output_layer
        return

    def check(self):
        if self.input_layer is None:
            print("input layer is None.")
            return False
        if self.output_layer is None:
            print("output layer is None.")
            return False

        if len(self.hidden_layers) < 1:
            print("hidden layers is empty.")
            return False
        return True

    def connect_layers(self):
        """set the input and output for the layers"""
        if not self.check():
            print("Failed to check nnetwork.")
            return

        # 1. set input layer
        pre_layer = self.input_layer
        for layer in self.hidden_layers:
            layer.set_input_layer(pre_layer)
            pre_layer = layer
        self.output_layer.set_input_layer(pre_layer)

        # 2. set output layer
        next_layer = self.output_layer
        for layer in reversed(self.hidden_layers):
            layer.set_next_layer(next_layer)
            next_layer = layer
        self.input_layer.set_next_layer(next_layer)

        return

    def forward(self, x):
        self.input_layer.feed(x)

        for layer in self.hidden_layers:
            layer.active()

        self.output_layer.active()
        return

    def backward(self, labels):
        self.output_layer.calc_error(labels)
        for layer in reversed(self.hidden_layers):
            layer.calc_error()

        return

    def update(self, lr):
        for layer in self.hidden_layers:
            layer.apply_delta(lr)
        self.output_layer.apply_delta(lr)
        return

    def train(self, x, y, lr):
        """train the model with single instance"""
        self.forward(x)
        self.backword(y)
        self.update(lr)
        return

    def __str__(self):
        msg = str(self.input_layer) + "\n"

        for layer in self.hidden_layers:
            msg += "%s\n" % (str(layer))

        msg += "%s\n" % (str(self.output_layer))
        return msg

    def get_detail(self):
        msg = "%s\n" % (self.input_layer.detail_info())

        for layer in self.hidden_layers:
            msg += ("%s\n" % (layer.detail_info()))

        msg += ("%s\n" % (self.output_layer.detail_info()))
        return msg


def test_nn():
    img_input = nn_layer.InputLayer("mnist_input", 784)
    output_layer = nn_layer.SoftmaxOutputLayer("mnist_output", 10)

    # 1. set input and output layers
    nn = NNetwork()
    nn.set_input(img_input)
    nn.set_output(output_layer)


    # 2. add some hidden layers
    h1 = nn_layer.HiddenLayer("h1", 256, activation.TanhActiveFunction)
    nn.add_hidden_layer(h1)

    h2 = nn_layer.HiddenLayer("h2", 256, activation.TanhActiveFunction)
    nn.add_hidden_layer(h2)

    # 3. print neural network
    print("%s" % (nn))

    # 4. complete nn construction
    nn.connect_layers()
    print(nn.get_detail())
    return


def main():
    print("hello")
    test_nn()
    return


if __name__ == "__main__":
    main()
