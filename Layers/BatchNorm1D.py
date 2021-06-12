import numpy as np

from Layers.BaseClasses.TrainableLayersABC import TrainableStandardLayersABC
from Layers.BaseClasses.OneDimLayer import OneDimStandardLayers


class BatchNorm1D(OneDimStandardLayers, TrainableStandardLayersABC):

    def __init__(self, input_length, batch_size, gamma=1, beta=0, eps=10 ** -5):
        OneDimStandardLayers.__init__(self, input_length=input_length, batch_size=batch_size)
        self._gamma = gamma * np.ones((input_length, 1))
        self._beta = beta * np.ones((input_length, 1))
        self._eps = eps * np.ones((input_length, 1))
        print(self._gamma)

    def forward_pass(self, input):
        self.calculate_input_var(input)
        self.calculate_normalised_input(input, self._input_var)
        return (self._normalised_input * self._gamma) + self._beta

    def backward_gradient(self, input, backwards_input):
        self.calculate_normalised_input_gradient(backwards_input)
        self.calculate_input_gradient(self._normalised_input_gradient)

    def calculate_normalised_input(self, input, input_var):
        self._normalised_input = (input - np.mean(input, axis=0)) / np.sqrt(input_var + self._eps)
        assert self._normalised_input.shape == (self._batch_size, self._input_length, 1)

    def calculate_input_var(self, input):
        self._input_var = np.var(input, axis=0)
        assert self._input_var.shape == (self._input_length, 1)

    def calculate_gamma_gradient(self, backwards_input):
        self._gamma_gradient = np.mean(backwards_input * self._normalised_input, axis=0)
        assert self._gamma_gradient.shape == (self._input_length, 1)

    def calculate_beta_gradient(self, backwards_input):
        self._beta_gradient = np.mean(backwards_input, axis=0)
        assert self._beta_gradient.shape == (self._input_length, 1)

    def calculate_normalised_input_gradient(self, backward_input):
        self._normalised_input_gradient = backward_input * self._gamma
        assert self._normalised_input_gradient.shape == (self._batch_size, self._input_length, 1)

    # def calculate_input_mean_gradient(self, input):
    #     # Based on the math found in https://kevinzakka.github.io/2016/09/14/batch_normalization/
    #     # TODO verify / work out math
    #     self._input_mean_gradient = np.mean(self._normalised_input * (-1 / np.sqrt(np.var(input) + self._eps)),
    #                                         axis=0)
    #     assert self._input_mean_gradient.shape == (self._input_length, 1)

    def calculate_input_gradient(self, normalised_input_gradient):
        self._input_gradient = (normalised_input_gradient - np.mean(normalised_input_gradient,
                                                                    axis=0) - self._normalised_input * np.mean(
            normalised_input_gradient * self._normalised_input)) / (self._input_var + self._eps)
        assert self._input_gradient.shape == (self._batch_size, self._input_length, 1)

    def calculate_weights_gradient(self, input, backwards_input):
        self.calculate_beta_gradient(backwards_input)
        self.calculate_gamma_gradient(backwards_input)

    def check_weights_gradient_dim(self):
        pass

    def init_weights(self, weight_init_method, bias_init_method):
        pass

    def update_weights(self, step_size):
        self._gamma = self._gamma-step_size*self._gamma_gradient
        self._beta = self._beta-step_size*self._beta_gradient


if __name__ == '__main__':
    print("Simple test of BatchNorm Layer")
    test_layer = BatchNorm1D(16, 40)
    input_data = np.random.randn(40, 16, 1)
    output_data = test_layer.forward(input_data)
    summed_data = np.sum(output_data, axis=0)
    print(summed_data)
    print(all(np.abs(summed_data) < 1))
    print(test_layer._gamma)

    test_layer.backwards(output_data)

    for i in range(10000):
        test_layer.train(1)

    print(test_layer.forward(input_data))
    print(test_layer._gamma)


