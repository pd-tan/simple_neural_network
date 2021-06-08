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

