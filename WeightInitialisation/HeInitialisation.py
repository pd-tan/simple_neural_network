import numpy as np

def HeInit(input_length,output_length):
    return np.random.randn(output_length,input_length)*np.sqrt(2/input_length)