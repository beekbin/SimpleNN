from __future__ import division
from __future__ import print_function
import struct
import os
import numpy as np
from datetime import datetime

import nn_layer
import activation


class NNetwork(object):
    def __init__(self):
        self.input_layer = None
        self.output_layer = None
        self.hidden_layers = []
        self.errors = []
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

        # 3. call layer init
        self.input_layer.init()
        for layer in self.hidden_layers:
            layer.init()
        self.output_layer.init()

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

        if len(self.errors) == 10000:
            print("cost=%.3f" % (sum(self.errors)/len(self.errors)))
            self.errors = []
        self.errors.append(self.output_layer.calc_cost(labels))
        return

    def update(self, lr):
        for layer in self.hidden_layers:
            layer.update_weights(lr)
        self.output_layer.update_weights(lr)
        return

    def train(self, x, y, lr):
        """train the model with single instance"""
        self.forward(x)
        self.backward(y)
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


def transform_label(label_1d, d2):
    num = label_1d.shape[0]
    result = np.zeros((num, d2))

    for i in range(num):
        label = label_1d[i]
        result[i, label] = 1
    return result


def load_data(path, dtype):
    if dtype == "train":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dtype == "test":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        print("[ERROR] dtype must be 'test' or 'train' vs. %s" % (dtype))
        return None

    # 1. load labels
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        labels = np.fromfile(flbl, dtype=np.int8)
        print("%s: magic=%d, number=%d" % (fname_lbl, magic, num))
        labels = transform_label(labels, 10)
        print(labels.shape)

    # 2. load images
    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        imgs = np.fromfile(fimg, dtype=np.uint8).reshape(-1, rows * cols)
        print("%s: magic=%d, num=%d, rows=%d, cols=%d" % (fname_img, magic, num, rows, cols))
        print(imgs.shape)

    return (labels, imgs)


def construct_nn():
    img_input = nn_layer.InputLayer("mnist_input", 784)
    output_layer = nn_layer.SoftmaxOutputLayer("mnist_output", 10)

    # 1. set input and output layers
    nn = NNetwork()
    nn.set_input(img_input)
    nn.set_output(output_layer)

    # 2. add some hidden layers
    h1 = nn_layer.HiddenLayer("h1", 256, activation.tanhFunc)
    nn.add_hidden_layer(h1)

    h2 = nn_layer.HiddenLayer("h2", 64, activation.tanhFunc)
    nn.add_hidden_layer(h2)

    h3 = nn_layer.HiddenLayer("h3", 10, activation.reluFunc)
    nn.add_hidden_layer(h3)

    # 3. complete nn construction
    #print("%s" % (nn))
    nn.connect_layers()
    print(nn.get_detail())
    return nn


def train_it(nn, train_data, lr):
    labels = train_data[0]
    imgs = train_data[1]

    for i in range(labels.shape[0]):
        label = labels[i, :]
        img = imgs[i, :]
        nn.train(img, label, lr)

    return


def train_nn(data_dir):
    nn = construct_nn()
    train_data = load_data(data_dir, "train")
    test_data = load_data(data_dir, "test")
    if (train_data is None) or (test_data is None):
        print("[ERROR] failed to load data")
        return

    lr = 0.005
    #train_it(nn, train_data, lr)
    for i in range(100):
        print("[%s] begin epo-%s" % (str(datetime.now()), i))
        if i > 5:
            lr = 0.001
        train_it(nn, train_data, lr)
        print("[%s] end epo-%s" % (str(datetime.now()), i))

    return


def main():
    #construct_nn()
    train_nn("./data/")
    return


if __name__ == "__main__":
    main()
