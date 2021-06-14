from Layers.BaseClasses.StandardLayersABC import StandardLayersABC
from abc import abstractmethod
class TrainableStandardLayersABC(StandardLayersABC):

    @abstractmethod
    def init_weights(self,weight_init_method,bias_init_method):
        pass

    @abstractmethod
    def _update_weights(self, step_size):
        pass

    @abstractmethod
    def _calculate_model_parameters_gradient(self, input, backwards_input):
        """
        Calculate the gradient of all model parameters to be updated during training.
        Args:
            input : The input used for current forward pass
            backwards_input : The backward input from downstream layer

        Returns:
            None

        """
        pass

    @abstractmethod
    def _check_weights_gradient_dim(self):
        pass

    def train(self,step_size):
        self._calculate_model_parameters_gradient(self._current_input, self._current_backward_input)
        self._check_weights_gradient_dim()
        self._update_weights(step_size)

