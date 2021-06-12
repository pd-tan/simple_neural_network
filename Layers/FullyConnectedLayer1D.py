import numpy as np
from WeightInitialisation.WeightInitialisationTypes import WeightInitialisationType, WeightInitialiser, ZeroInit, \
    HeInit, RandomInit
from Layers.BaseClasses.TrainableLayersABC import TrainableStandardLayersABC
from Layers.BaseClasses.OneDimLayer import OneDimStandardLayers


class FullyConnectedLayer1D(OneDimStandardLayers, TrainableStandardLayersABC):
    def __init__(self, input_length, output_length, batch_size, weight_init_method=WeightInitialisationType.HE):
        OneDimStandardLayers.__init__(self, input_length=input_length, batch_size=batch_size)
        self._output_length = output_length

        self.init_weights(weight_init_method)
        self._initial_weights = self._weights

    def forward_pass(self, input):
        return np.matmul(self._weights, input) + self._biases

    def backward_gradient(self, input, backwards_input):
        return np.expand_dims(np.sum(backwards_input*self._weights,axis=1),axis=2)


    def calculate_weights_gradient(self, input, backwards_input):
        self.calculate_w_gradient(input, backwards_input)
        self.calculate_b_gradient(backwards_input)

    def calculate_w_gradient(self, input, backwards_input):
        # assuming entire minibatch train in one step.
        self._weights_gradient = np.mean(np.matmul(backwards_input, np.swapaxes(input, 1, 2)), axis=0)

    def calculate_b_gradient(self, backwards_input):
        # assuming entire minibatch train in one step.
        self._biases_gradient = np.mean(backwards_input, axis=0)

    def check_weights_gradient_dim(self):
        self.check_w_dim()
        self.check_b_wim()

    def check_w_dim(self):
        assert self._weights_gradient.shape == self._weights.shape

    def check_b_wim(self):
        assert self._biases_gradient.shape == self._biases.shape

    def init_weights(self, weight_init_method, bias_init_method=None):
        self._weights = WeightInitialiser(init_type=weight_init_method, input_length=self._input_length,
                                          output_length=self._output_length, batch_size=self._batch_size)
        if bias_init_method == None:
            self._biases = np.zeros((self._output_length, 1))

    def update_weights(self, step_size):
        # simple linear regression
        self._weights = self._weights - step_size * self._weights_gradient
        self._biases = self._biases - step_size * self._biases_gradient

    def has_weights(self):
        return True


if __name__ == '__main__':
    print("Simple test of FC1D Layer");
    test_layer = FullyConnectedLayer1D(3, 2, 5);
    input = np.random.randn(5, 3, 1)
    print(input.shape)
    print(test_layer.forward(np.random.randn(5, 3, 1)).shape)


    from data_generation.generate_data import data_generator
    from CrossEntropyLoss import CrossEntropyLoss
    from SigmoidLayer import SigmoidLayer
    from WeightInitialisation.WeightInitialisationTypes import WeightInitialisationType
    number_of_features_ = 4
    batch_size = 10
    number_of_runs = 10000000
    print_per_runs = 10000
    generator = data_generator(4)
    test_layer = FullyConnectedLayer1D(number_of_features_,2,batch_size)
    test_layer.init_weights(weight_init_method=WeightInitialisationType.HE)
    sigmoid_layer = SigmoidLayer(2,batch_size)
    CELoss = CrossEntropyLoss(2,batch_size)
    step_size = 0.00001
    loss_sum = 0
    for run in range(number_of_runs):
        # print("run:",run)
        data,truth = generator.generate_data(batch_size=batch_size)
        loss = CELoss.forward(sigmoid_layer.forward(test_layer.forward(data)),truth)
        test_layer.backwards(sigmoid_layer.backwards(CELoss.backwards()))
        test_layer.train(step_size)
        # print(test_layer._weights)
        loss_sum = loss_sum+loss
        if run%print_per_runs==0:
            print("progress:", run/number_of_runs)
            print(loss_sum/print_per_runs)
            loss_sum = 0
            print(generator._weightage_vector)
            print(test_layer._initial_weights)
            print(test_layer._weights)



