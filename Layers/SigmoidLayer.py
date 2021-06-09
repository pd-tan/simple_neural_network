from Layers.BaseClasses.LayersABC import LayerABC
import numpy as np
from Layers.BaseClasses.OneDimLayer import OneDimLayer

class SigmoidLayer(OneDimLayer):
    def __init__(self,input_length,batch_size):
        super().__init__(input_length=input_length,batch_size=batch_size)
        pass

    def forward(self, input):
        super().forward(input)
        return 1 / (1 + np.exp(-input))

    def back(self):
        pass


if __name__ == '__main__':
    print("Simple test of sigmoid Layer");
    test_layer = SigmoidLayer(3,10)
    print(test_layer.forward(np.random.randn(10,3,1)))
