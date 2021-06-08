from layers.LayersABC import LayerABC
import numpy as np

class ReLU1D(LayerABC):
    def __init__(self):


    def forward(self, input):
        return input * (input > 1)

    def back(self, input):
        # TODO add backwards for relu
        pass



if __name__ == '__main__':
    print("Simple test of ReLU1D Layer");
    test_layer = ReLU1D()
    print(test_layer.forward(np.random.rand(16, 1)))
