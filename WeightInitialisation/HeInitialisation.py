import numpy as np

def HeInit(input_length,output_length,batch_size):
    return np.random.randn(batch_size,output_length,input_length)*np.sqrt(2/input_length)