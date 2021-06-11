import numpy as np

from Layers.BaseClasses.LossFunctionLayerABC import LossLayersABC


class CrossEntropyLoss(LossLayersABC):
    def __init__(self, input_length, batch_size):
        assert (input_length != 0), "Input must not be 0"
        assert (batch_size != 0), "Batch size must not be 0"
        self._input_length = input_length
        self._batch_size = batch_size

    def backward_gradient(self, input, truth):
        return -(truth / input + (1 - truth) * -1 / (1 - input))

    def forward_pass(self, input, truth):
        loss_ = -(truth * np.log(input) + (1 - truth) * np.log(1 - input))
        loss_ = np.sum(loss_, axis=1)
        return loss_

    def check_input_dim(self, input):
        assert (len(input.shape) == 3), "Input must be one-dimensional x batch size"
        assert (input.shape[
                    0] == self._batch_size), "The first dimension of your input should be the speciified batch size"
        assert (input.shape[
                    1] == self._input_length), "The second dimension of your input should be the input length specified"
        assert (input.shape[2] == 1), "This is a 1D layer, the third dimension of your input should be 1"

    def check_truth_dim(self, truth):
        assert (len(truth.shape) == 3), "Ground truth must be one-dimensional x batch size"
        assert (truth.shape[
                    0] == self._batch_size), "The first dimension of your Ground truth  should be the speciified batch size"
        assert (truth.shape[
                    1] == self._input_length), "The second dimension of your Ground truth  should be the input length specified"
        assert (truth.shape[2] == 1), "This is a 1D layer, the third dimension of your Ground truth  should be 1"


if __name__ == '__main__':
    from Layers.SigmoidLayer import SigmoidLayer

    print("Simple test of CELoss Layer");
    test_layer = CrossEntropyLoss(3, 10)
    sigmod = SigmoidLayer(3, 10)
    network_output = np.random.randn(10, 3, 1)
    truth = np.round(np.random.uniform(0, 1, (10, 3, 1)))
    print(sigmod.forward(network_output))
    print(truth)
    loss = test_layer.forward(sigmod.forward(network_output), truth)
    print(loss)
    print(test_layer.backwards())
