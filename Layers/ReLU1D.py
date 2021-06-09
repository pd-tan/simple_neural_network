from Layers.BaseClasses.LayersABC import LayerABC
from Layers.BaseClasses.OneDimLayer import OneDimLayer
import numpy as np


class ReLU1D(OneDimLayer):
    def __init__(self,input_length,batch_size):
        super().__init__(input_length=input_length,batch_size=batch_size)
        pass

    def forward(self, input):
        super().forward(input)
        return input * (input > 0)

    def back(self, input):
        # TODO add backwards for relu
        pass


if __name__ == '__main__':
    print("Simple test of ReLU1D Layer");
    test_layer = ReLU1D(3,16)
    print(test_layer.forward(np.random.randn(16,3,1)))
