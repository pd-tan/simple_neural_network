from Layers.BaseClasses.StandardLayersABC import StandardLayersABC
import numpy as np
from Layers.BaseClasses.OneDimLayer import OneDimStandardLayers


class SigmoidLayer(OneDimStandardLayers):
    def __init__(self, input_length, batch_size):
        super().__init__(input_length=input_length, batch_size=batch_size)

        pass

    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    def forward_pass(self, input):
        return self.sigmoid(input=input)

    def backward_gradient(self, input, backwards_input):
        return self.sigmoid(input) * (1 - self.sigmoid(input)) * backwards_input


if __name__ == '__main__':
    print("Simple test of sigmoid Layer");
    test_layer = SigmoidLayer(3, 10)
    print(test_layer.forward(np.random.randn(10, 3, 1)))
    print(test_layer.backwards(np.random.randn(10, 3, 1)))
