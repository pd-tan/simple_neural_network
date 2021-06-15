from Layers.BaseClasses.StandardLayersABC import StandardLayersABC
from abc import abstractmethod
class TrainableStandardLayersABC(StandardLayersABC):

    @abstractmethod
    def _init_trainable_parameters(self, weight_init_method, bias_init_method):
        """

        Args:
            weight_init_method:
            bias_init_method:

        Returns:
            None

        """
        pass

    @abstractmethod
    def _update_trainable_parameters(self, step_size):
        """
        Update parameters of the layer. Current only simple linear regression is supported
        Args:
            step_size (float): Updates the parameter by step_size * gradient

        Returns:
            None
        Todo:
            * support different training regimes
        """

        pass

    @abstractmethod
    def _calculate_trainable_parameters_gradient(self, input, backwards_input):
        """
        Calculate the gradient of all model parameters to be updated during training. Saves gradient as variable
        Args:
            input : The input used for current forward pass
            backwards_input : The backward input from downstream layer

        Returns:
            None

        """
        pass

    @abstractmethod
    def _check_trainable_parameters_gradient_dim(self):
        """
        Checks the dimensions of the gradient calculated and saved.
        Returns:
             None
        Raises:
            AssertionError: If dimension of gradient differes from required dimension

        """

        pass

    def train(self,step_size):
        self._calculate_trainable_parameters_gradient(self._current_forward_input, self._current_backward_input)
        self._check_trainable_parameters_gradient_dim()
        self._update_trainable_parameters(step_size)

