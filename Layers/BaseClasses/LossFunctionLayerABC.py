from abc import ABCMeta, abstractmethod


class LossLayersABC(metaclass=ABCMeta):

    @abstractmethod
    def _get_backward_output(self, forward_input, truth):
        """
        Calculates the backward output to pass to previous layer during back propogation
        Args:
            forward_input: Input used in this forward pass (previous layer forward output)
            truth: The true value used in this forward pass to calculate loss

        Returns:
            backward_output

        """
        pass

    @abstractmethod
    def _forward_pass(self, forward_input, truth):
        pass

    @abstractmethod
    def _check_forward_input_dim(self, forward_input):
        """
        Check dimension of forward input based on batch size and input dimension set during construction
        Args:
            forward_input:

        Raises:
            AssertionError: If dimension of forward input does not agree with required dimension.

        """
        pass

    @abstractmethod
    def _check_truth_dim(self, truth):
        """
        Check dimension of truth based on batch size and input dimension set during construction
        Args:
            truth:

        Raises:
            AssertionError: If dimension of truth does not agree with required dimension.

        """
        pass

    def _save_current_forward_input(self, forward_input):
        """
        Saves current forward input from previous layer to be used in calculation of gradients
        Args:
            forward_input:

        Returns:
            None

        """
        self._current_forward_input = forward_input

    def _save_current_truth(self, current_truth):
        """
        Saves the current truth values used for forward pass for future calculations
        Args:
            current_truth: Array of the values True label for calculation of loss.

        Returns:
            None

        """
        self._current_truth = current_truth

    def forward(self, input,truth):
        self._check_forward_input_dim(forward_input=input)
        self._check_truth_dim(truth=input)
        self._save_current_forward_input(input)
        self._save_current_truth(truth)
        return self._forward_pass(input, truth)

    def backwards(self):
        return self._get_backward_output(forward_input=self._current_forward_input, truth=self._current_truth)

