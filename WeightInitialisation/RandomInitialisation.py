import numpy as np

def RandomInit(input_length,output_length,batch_size):
    return np.random.randn(batch_size,output_length,input_length)