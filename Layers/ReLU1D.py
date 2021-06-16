from Layers.BaseClasses.StandardLayersABC import StandardLayersABC
from Layers.BaseClasses.OneDimLayer import OneDimStandardLayers
import numpy as np


class ReLU1D(OneDimStandardLayers):
    def __init__(self,input_length,batch_size):
        super().__init__(input_length=input_length,batch_size=batch_size,output_length=input_length)
        pass

    def _forward_pass(self, forward_input):
        return forward_input * (forward_input > 0)

    def _get_backward_output(self, forward_input, backwards_input):
        return 1 * (forward_input > 0) * backwards_input


if __name__ == '__main__':
    print("Simple test of ReLU1D Layer");
    test_layer = ReLU1D(3,16)
    print(test_layer.forward(np.random.randn(16,3,1)))
    print(test_layer.forward(np.random.randn(16,3,1)))
