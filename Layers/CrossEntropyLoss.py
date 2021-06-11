import numpy as np

from Layers.BaseClasses.OneDimLayer import OneDimLayer


class CrossEntropyLoss(OneDimLayer):
    def __init__(self, input_length, batch_size):
        super().__init__(input_length=input_length, batch_size=batch_size)
        pass
    def cross_entropy_loss(self, input, truth):
        loss = -(truth * np.log(input) + (1 - truth) * np.log(1 - input))
        loss = np.sum(loss, axis=1)
        return loss

    def gradient(self,input,backwards_input):
        return -(backwards_input / input + (1 - backwards_input) * -1 / (1 - input))

    def forward(self, input, truth):
        self.check_input_dim(input)
        self.check_input_dim(truth)
        self.save_current_input(input)
        self.save_current_truth(truth)
        return self.cross_entropy_loss(input=input, truth=truth)


    def save_current_truth(self, truth):
        # TODO consider a superclass for loss function and have this defined in it.
        self._current_truth = truth

    def back(self, backwards_input=None):
        return self.cross_entropy_loss_backward(self._current_input,self._current_truth)


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
    print(test_layer.back())
