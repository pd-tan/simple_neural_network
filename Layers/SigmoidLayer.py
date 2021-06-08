from Layers.BaseClasses.LayersABC import LayerABC
import numpy as np
from Layers.BaseClasses.OneDimLayer import OneDimLayer

class SigmoidLayer(OneDimLayer):
    def __init__(self,input_length):
        super().__init__(input_length=input_length)
        pass

    def forward(self, input):
        return 1 / (1 + np.exp(-input))

    def back(self):
        pass


if __name__ == '__main__':
    print("Simple test of sigmoid Layer");
    test_layer = SigmoidLayer(3)
    print(test_layer.forward(np.array([100,0,-100])))
