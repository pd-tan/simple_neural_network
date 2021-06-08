import numpy
import numpy as np

from layers.layers_abc import LayerABC


class FullConnectedLayer1D(LayerABC):
    def __init__(self, input_length=16, output_length=16, init_method=None):
        self._input_length = input_length
        self._output_length = output_length
        self.init_weights(init_method)

    def forward(self, input):
        return np.matmul(self._weights, input)

    def back(self):
        # TODO implement backwards
        pass

    def init_weights(self, init_method=None):
        if init_method == None:
            self._weights = np.random.rand(self._output_length, self._input_length)
        else:
            pass

    def has_weights(self):
        return True


if __name__ == '__main__':
    print("Simple test of FC1D Layer");
    test_layer = FullConnectedLayer1D();
    print(test_layer.forward(np.random.rand(16,1)))
