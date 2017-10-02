from __future__ import print_function

import numpy as np
import sys
import random
import logging
import os

sys.path.insert(0, "../")
from nn.embedding_layer import EmbeddingLayer
from nn.nn_layer import InputLayer
from nn.nn_layer import ActiveLayer
from nn.simple_nn import NNetwork


class FakeOutputLayer(ActiveLayer):
    def __init__(self, name, size):
        super(FakeOutputLayer, self).__init__(name, size)
        return

    def calc_input_delta(self, output):
        np.copyto(output, self.delta)
        return

    def active(self):
        self.output = self.input_layer.get_output()
        return

    def calc_error(self, labels):
        self.delta = labels - self.output
        return

    def calc_cost(self, labels):
        cost = np.sum(np.absolute(self.delta))
        return cost


def get_random_vectors(m, n):
    result = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        result[i] = np.random.uniform(-1, 1, n)
    return result


def train_it(nn, y, m):
    data = []
    for i in range(100):
        x = random.randint(0, m-1)
        data.append(x)

    lr = 0.001
    for x in data:
        nn.train(x, y[x], lr)
    return


def test1():
    m = 3
    n = 4
    y = get_random_vectors(m, n)

    nn = NNetwork()
    myin = InputLayer("input", 1)
    emb = EmbeddingLayer("emb1", m, n)
    myout = FakeOutputLayer("output", n)

    nn.set_input(myin)
    nn.set_output(myout)
    nn.add_hidden_layer(emb)
    nn.connect_layers()
    nn.set_log_interval(100)

    for i in range(1000):
        train_it(nn, y, m)

    print(y)
    print("*"*40)
    print(emb.weights)
    return


def main():
    test1()
    return 0


def setup_log():
    logfile = "./test.emb.%s.log" % (os.getpid())
    if len(sys.argv) > 1:
        logfile = sys.argv[1]
    print("logfile=%s" % (logfile,))
    logging.basicConfig(filename=logfile, format='[%(asctime)s] %(message)s', level=logging.DEBUG)
    return


if __name__ == "__main__":
    sys.exit(main())
