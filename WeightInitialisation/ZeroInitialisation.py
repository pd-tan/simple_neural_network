import numpy as np

def ZeroInit(input_length,output_length,batch_size):
    return np.zeros((batch_size,output_length,input_length))