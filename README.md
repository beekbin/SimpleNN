# SimpleNN: a simple neural network framework
This project builds a simple neural network framework based on python and Numpy.

With this framework, a simple 5-layers neural network is built. The [main.py](https://github.com/beekbin/SimpleNN/blob/master/main.py) file demonstrates how to use the simple framework to build the NN, and how to train the NN with `MNIST` dataset. (`CNN` version can be found in another [project](https://github.com/beekbin/simpleCNN).)


# Achieve 98% correctness on MNIST
Here is to describe how to use this simple framework to build a neural network, and how to train it on `MNIST` dataset to achieve 98% correctness. 

Even though the network is simple, and the number is not [impressive at all](mnist), it is still interesting to practice the common optimization skills to train a neural network.


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

## data normalization
This is the most important adjustment after I finished the code.
Before doing data normalization, I can hardly achieve 70% correctness(cost was around 0.70) on the MNIST dataset. After a very simple normalization is applied to the input data, 95% correctness is achieved immediately in the first epoch.
```python
def normalize_img(imgs):
    result = imgs.astype(float)
    result /= 255.0
    avg = np.average(result, axis=1).reshape((-1, 1))
    result -= avg
    return result
```

## data shuffling
Because the naive SGD is used during the training, it is also important to shuffle the data during training.
```python
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

# Run it
### prerequisites
    * Python 2.7+
    * Numpy

### run it
 ```console
 # 1. get code
 git clone https://github.com/beekbin/SimpleNN.git
 cd SimpleNN
 
 # 2. download the mnist data
 cd data
 sh get.sh
 cd ..
 
 # 3. run it
 python main.py
 ```

# TODO

### 1. Add more kinds of layers
Only fully connected layers are supported in current implementation.

**Convolutional + Pooling Layers**

    This is implemented in [another project](https://github.com/beekbin/simpleCNN).
    
**Embedding Layer**
   This is important for NLP problems. 
   
**Recurrent Layers**
   such as vanilla RNN, LSTM, GRU.   

### 2. Improve generalization
**Dropout**

### 3. Accelerate training process
**BatchNorm**

**Adaptive learning rate schedulers**

   Here is a wonderful review of the popular [learning rate schdulers](http://ruder.io/optimizing-gradient-descent/),
   I'd like to try some of them.
   

### 4. More flexible layers: allow multiple inputs

 In current implementation, one layer can only have one input layer. However, many modern deep learning networks requries 
 multiple inputs, such as [ResNet](https://arxiv.org/abs/1512.03385)/[HighwayNet](https://arxiv.org/abs/1505.00387)/[DenseCNN](https://arxiv.org/abs/1608.06993).  And in one of my project, we found that even Densely connected LSTM layers are also powerful.
 
