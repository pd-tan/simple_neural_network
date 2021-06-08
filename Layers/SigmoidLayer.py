from Layers.LayersABC import LayerABC
import numpy as np


class SigmoidLayer(LayerABC):
    def __init__(self):
        pass

    def forward(self, input):
        return 1 / (1 + np.exp(-input))

    def back(self):
        pass


if __name__ == '__main__':
    print("Simple test of sigmoid Layer");
    test_layer = SigmoidLayer()
    print(test_layer.forward(np.array([100,0,-100])))
