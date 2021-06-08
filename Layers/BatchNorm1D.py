import numpy as np

from Layers.BaseClasses.LayersABC import LayerABC
from Layers.BaseClasses.OneDimLayer import OneDimLayer

class BatchNorm1D(OneDimLayer):

    def __init__(self, input_length, gamma=1, beta=0, eps=10 ** -5):
        OneDimLayer.__init__(self,input_length=input_length)
        self._gamma = gamma
        self._beta = beta
        self._eps = eps

    def forward(self, input):
        # TODO asset dim
        return((input - np.mean(input)) * self._gamma / np.sqrt(np.var(input) + self._eps)) + self._beta

    def back(self):
        # TODO implement backwareds for batchnorm

        pass

if __name__ == '__main__':
    print("Simple test of BatchNorm Layer");
    test_layer = BatchNorm1D(16)
    print(test_layer.forward(np.random.rand(16, 1) * 3))
