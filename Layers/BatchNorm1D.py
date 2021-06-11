import numpy as np

from Layers.BaseClasses.StandardLayersABC import StandardLayersABC
from Layers.BaseClasses.OneDimLayer import OneDimStandardLayers

class BatchNorm1D(OneDimStandardLayers):

    def __init__(self, input_length, batch_size, gamma=1, beta=0, eps=10 ** -5):
        OneDimStandardLayers.__init__(self, input_length=input_length, batch_size=batch_size)
        self._gamma = gamma
        self._beta = beta
        self._eps = eps

    def forward_pass(self, input):
        return((input - np.mean(input,axis=0)) * self._gamma / np.sqrt(np.var(input) + self._eps)) + self._beta

    def backward_gradient(self, input,backwards_input):
        pass

if __name__ == '__main__':
    print("Simple test of BatchNorm Layer");
    test_layer = BatchNorm1D(16,40)
    output_data = test_layer.forward(np.random.randn(40,16,1))
    summed_data = np.sum(output_data,axis=0)
    print(summed_data)

    print(all(np.abs(summed_data)<1))

