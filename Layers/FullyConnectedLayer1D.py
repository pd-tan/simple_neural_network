import numpy as np
from WeightInitialisation.WeightInitialisationTypes import WeightInitialisationType, WeightInitialiser, ZeroInit, \
    HeInit, RandomInit
from Layers.BaseClasses.TrainableLayersABC import TrainableLayersABC
from Layers.BaseClasses.OneDimLayer import OneDimLayer


class FullyConnectedLayer1D(OneDimLayer, TrainableLayersABC):
    def __init__(self, input_length, output_length, batch_size, weight_init_method=WeightInitialisationType.ZERO):
        OneDimLayer.__init__(self, input_length=input_length, batch_size=batch_size)
        self._output_length = output_length
        self.init_weights(weight_init_method)

    def forward(self, input):
        super().check_input_dim(input)

        return np.matmul(self._weights, input) + self._biases

    def back(self,input,backwards_input):
        return backwards_input*np.expand_dims(np.sum(self._weights,axis=0),axis=1)


    def init_weights(self, weight_init_method, bias_init_method=None):
        self._weights = WeightInitialiser(init_type=weight_init_method, input_length=self._input_length,
                                          output_length=self._output_length, batch_size=self._batch_size)
        if bias_init_method == None:

            self._biases = np.zeros((self._output_length,1))

    def update_weight(self):
        # TODO implement update weights
        pass

    def has_weights(self):
        return True


if __name__ == '__main__':
    print("Simple test of FC1D Layer");
    test_layer = FullyConnectedLayer1D(3,2,5);
    input = np.random.randn(5,3,1)
    print(input.shape)
    print(test_layer.forward(np.random.randn(5,3,1)).shape)
