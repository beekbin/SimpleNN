from __future__ import print_function
from __future__ import division
import numpy as np
from nn_layer import Layer
import logging


class EmbeddingLayer(Layer):
    def __init__(self, name, num, dim):
        """num: number of word;
           dim: dimension of the word vector;
        """
        super(EmbeddingLayer, self).__init__(name, num * dim)
        self.num = num
        self.dim = dim
        self.weights = None
        self.word_index = -1
        self.delta = None
        return

    def init(self):
        self.weights = np.zeros((self.num, self.dim), dtype=np.float64)
        for i in range(self.num):
            self.weights[i] = np.random.uniform(-1, 1, self.dim)
        self.delta = np.zeros(self.dim, dtype=np.float64)
        return

    def active(self):
        indata = self.input_layer.get_output()

        if type(indata) is not int:
            logging.error("input of EmbeddingLayer[%s] should be a integer." % (self.name,))
            return

        if indata > self.num:
            logging.error("EmbeddingLayer[%s] out of index [%s Vs. %s]." % (self.name, indata, self.num))
            return
        self.word_index = indata
        self.output = self.weights[indata]
        return

    def calc_error(self):
        self.next_layer.calc_input_delta(self.delta)
        return

    def update_weights(self, lr):
        if self.lambda2 > 0:
            self.delta += (self.lambda2 * self.weights)

        self.weights -= lr * self.delta
        return
