from abc import ABCMeta, abstractmethod


class StandardLayersABC(metaclass=ABCMeta):

    @abstractmethod
    def _get_backward_output(self, forward_input, backwards_input):
        """

        Args:
            forward_input: The input used for current forward pass
            backwards_input : The backward input from downstream layer

        Returns:
            backward_output: The output for previous layers to continue back propagation

        """
        pass

    @abstractmethod
    def _forward_pass(self, forward_input):
        """

        Args:
            forward_input: The input used for current forward pass

        Returns:
            forward_output: The output for next layer to use as input in forward pass

        """
        pass

    @abstractmethod
    def _check_input_dim(self, forward_input):
        """
        Check dimension of forward input based on batch size and input dimension set during construction
        Args:
            forward_input:

        Raises:
            AssertionError: If dimension of forward input does not agree with required dimension.

        """
        pass

    @abstractmethod
    def _check_backwards_input_dim(self, backwards_input):
        """
        Check dimension of backward input based on batch size and input dimension set during construction
        Args:
            backwards_input:

        Returns:
            AssertionError: If dimension of backward input does not agree with required dimension.

        """
        pass

    def _save_current_input(self, forward_input):
        self._current_forward_input = forward_input


    def _save_current_backward_input(self, backward_input):
        self._current_backward_input = backward_input

    def forward(self, forward_input):
        self._check_input_dim(forward_input=forward_input)
        self._save_current_input(forward_input)
        return self._forward_pass(forward_input)

    def backwards(self, backwards_input):
        self._save_current_backward_input(backward_input=backwards_input)
        return self._get_backward_output(forward_input=self._current_forward_input, backwards_input=backwards_input)

