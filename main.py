from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import struct
import os
import numpy as np
from datetime import datetime

from nn import nn_layer
from nn import activation
from nn import simple_nn


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
    nn = simple_nn.NNetwork()
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