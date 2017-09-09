p.random.randn(d1, d2)


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
        self.delta_weights = np.zeros(self.fan_in, self.size)
        return

    def active(self):
        pass

    def calc_error(self):
        pass

    def get_sigma(self):
        return self.sigma

    def update_weights(self, lr):
        self.weights -= lr * self.delta_weights
        self.bias -= lr * self.sigma
        return


class SoftmaxOutputLayer(ActiveLayer):
    def __init__(self, name, size):
        super(SoftmaxOutputLayer, self).__init__(name, size)
        return

    def active(self):
        x = self.input_layer.get_output()
        np.dot(x, self.weights, out=self.z)
        self.z += self.bias

        # TODO: pass self.output into the function
        self.output = calc_softmax(self.z)
        return

    def calc_error(self, labels):
        self.sigma = self.output - labels

        x = self.input_layer.get_output()
        np.dot(x.reshape((-1, 1)), self.sigma.reshape((1, -1)), out=self.delta_weights)
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
        return

    def calc_error(self):
        # 1. calc sigma
        next_sigma = self.next_layer.get_sigma()
        np.dot(self.weights, next_sigma, out=self.sigma)
        self.sigma = self.sigma * self.func.backward(self.z)

        # 2. calc delta_weights
        x = self.input_layer.get_output()
        self.delta_weights = np.dot(x.reshape((-1, 1)), self.sigma.reshape((1, -1)))
        return
