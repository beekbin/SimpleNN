# SimpleNN
A simple Neural Network is able to achieve 98.11% correct rate on MNIST dataset. It is simple because it has not covoluational layer, nor RNN layer. 

Even though the network is simple, and the number is not impressive at all, it is still valuable after learning the optimization skills by [reading](https://arxiv.org/abs/1206.5533) and [learning](https://www.coursera.org/learn/neural-networks/home/welcome).



# Achieve 98.11% correctness
Here is to describe how to use this simple nerual network to achieve 98.11% correctness.

## data normalization
This is the most important adjustment after I finished the code.
Before doing data normalization, I can hardly achieve 70% correctness(cost was around 0.70) on the MNIST dataset. After a very simple normalization is applied to the input data, 95% correctness is achieved immediately.
```python
def normalize_img(imgs):
    result = imgs.astype(float)

    # convert data from [0, 255] to [0, 1.0]
    result /= 255.0
    avg = np.average(result, axis=1).reshape((-1, 1))
    result -= avg
    return result
```


## Nerual Network structure
This neural network has three hidden layers, and a softmax output layer.
The first two hidden layers uses __tanh__ activation function, the third hidden layer uses the __Relu__ activation function.
The derivative of __Relu__ function is 1, so it can transmit the errors from output to previous layers with less loss.

```python
def construct_nn(l2=0.0):
    img_input = nn_layer.InputLayer("mnist_input", 784)
    output_layer = nn_layer.SoftmaxOutputLayer("mnist_output", 10)

    # 1. set input and output layers
    nn = simple_nn.NNetwork()
    nn.set_input(img_input)
    nn.set_output(output_layer)

    # 2. add some hidden layers
    h1 = nn_layer.HiddenLayer("h1", 256, activation.tanhFunc)
    h1.set_lambda2(l2)
    nn.add_hidden_layer(h1)

    h2 = nn_layer.HiddenLayer("h2", 64, activation.tanhFunc)
    h2.set_lambda2(l2)
    nn.add_hidden_layer(h2)

    h3 = nn_layer.HiddenLayer("h3", 10, activation.reluFunc)
    h3.set_lambda2(l2)
    nn.add_hidden_layer(h3)

    # 3. complete nn construction
    nn.connect_layers()
    print(nn.get_detail())
    return nn
```

With this nerual network (l2=0.0), and simple SGD (no mini-batch), 10 epochs achieved 98.11% correctness on the test dataset.
```console
[train] accuracy=0.9972, avg_cost=0.0148
[test] accuracy=0.9811, avg_cost=0.0656
```

## data shuffling
Because the naive SGD is used during the training, it is also important to shuffle the data during training.
```

def train_it(nn, train_data, lr):
    labels = train_data[0]
    imgs = train_data[1]

    # shuffle the data
    alist = range(labels.shape[0])
    shuffle(alist)

    for i in alist:
        label = labels[i, :]
        img = imgs[i, :]
        nn.train(img, label, lr)

    return
```
