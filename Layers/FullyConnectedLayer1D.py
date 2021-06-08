import numpy as np

from Layers.BaseClasses.TrainableLayersABC import TrainableLayersABC
from Layers.BaseClasses.OneDimLayer import OneDimLayer

class FullyConnectedLayer1D(OneDimLayer,TrainableLayersABC):
    def __init__(self, input_length, output_length, init_method=None):
        OneDimLayer.__init__(self,input_length=input_length)
        self._output_length = output_length
        self.init_weights(init_method)

    def forward(self, input):
        super().forward(input)
        return np.matmul(self._weights, input) + self._biases

    def back(self):
        # TODO implement backwards
        pass

    def init_weights(self, init_method=None):
        if init_method == None:
            self._weights = np.random.rand(self._output_length, self._input_length)
            self._biases =np.random.rand(self._output_length)
        else:
            pass

    def update_weight(self):
        #TODO implement update weights
        pass

    def has_weights(self):
        return True


if __name__ == '__main__':
    print("Simple test of FC1D Layer");
    test_layer = FullyConnectedLayer1D(1,16);
    print(test_layer.forward(np.random.rand(1)))
