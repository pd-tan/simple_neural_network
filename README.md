# simple_neural_network
A simple neural netwoprk implemented in python using numpy


# Thought Process

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
