import numpy as np

from Layers.BaseClasses.LayersABC import LayerABC
from Layers.BaseClasses.OneDimLayer import OneDimLayer

class BatchNorm1D(OneDimLayer):

    def __init__(self, input_length, batch_size, gamma=1, beta=0, eps=10 ** -5):
        OneDimLayer.__init__(self,input_length=input_length,batch_size=batch_size)
        self._gamma = gamma
        self._beta = beta
        self._eps = eps

    def forward(self, input):
        # TODO norm for each batch seperately
        super().forward(input)
        return((input - np.mean(input,axis=0)) * self._gamma / np.sqrt(np.var(input) + self._eps)) + self._beta

    def back(self):
        # TODO implement backwareds for batchnorm

        pass

if __name__ == '__main__':
    print("Simple test of BatchNorm Layer");
    test_layer = BatchNorm1D(16,4)
    print(test_layer.forward(np.random.randn(4,16,1)))
