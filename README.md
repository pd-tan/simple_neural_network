# Simple Neural Network

In this repository, a simple neural network is implemented using numpy. This library has been made to support a 3 layer
network structure

As such the library currently only support the following layers

1. BatchNorm
2. Fully Connected
3. ReLU
4. Sigmoid

## Sample implementation

### Loss function

The network shown above was trained and implemented. Based on the argmax layer at the end, it was assumed that this
layer was made for a binary classification task between two classes. As such the network was trained using a Binary
Cross Entropy loss function

### Data Generation

A [data generator](DataGeneration/DataGenerator.py) was developed to create a simple set of data for testing the framework. Upon initialisation, the data
generator would create a `(batch_size,input_length,1)` array known as the _template array_. The values in the template
array are obtained randomly using a normal distribution

At each call of data generation, a random `(input_length,1)`  array is created (normally idstributed) to simulate an
input.

A _truth_ array that is of shape `(batch_size,2,1)`

For each batch, the dot product between the _input_ and _template array_ is taken. Since both arrays are normally
distributed with a mean of 0, if the dot product is greater than 0, the _output_ is an array of `[[1],[0]]` else _
output_ is `[[0],[1]]`

### Training

Using the data generated and a simple linear regression for training, the following results were obtained.
![Training Image](images/img.png)

As expected, with a well defined true function, the neural network very quickly and easily converges to model the
function, even when running only on CPU.

This script can be found in [here](Network/simple_test.py)

## Weight Initialisation for Fully Connected Layer

Three forms of weight initialisation were developed:

1. [Zero](WeightInitialisation/ZeroInitialisation.py)
2. [Normal](WeightInitialisation/RandomInitialisation.py)
3. [He](WeightInitialisation/HeInitialisation.py)

### Zero Initialisation

This initialisation method is not to be used for an actual network. This was developed for future uses, such as in
implementing unit test code

### Normal Initialisation

A simple standard normal distribution is used to initialised the weights.

### He Initialisation

He initialisation was implemented as the preferred intialisation method for this network as ReLU activation was used in
the network. The benefits of He initialisation has been shown empirically and also shown theoretically to work better

## Layer ABC

'public' fucntions

- forward
- backwards
- has_weights
- init_weights(init_method(enum?))

'protected' functions

- set_weights
- get_gradient

'private' variable

- weights

## FC layer : Layer ABC subclass

- FC(input_nodes, output_nodes)
- weights
- biases

# TODO

1. Implement batch support
1. Test batch norm
2. Implement backprop
3. create network class that has ordered list of layers. Iterate all forward and backwards for convenience
