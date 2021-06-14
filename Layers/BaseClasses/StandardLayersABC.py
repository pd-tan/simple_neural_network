from abc import ABCMeta, abstractmethod


class StandardLayersABC(metaclass=ABCMeta):

    @abstractmethod
    def _backward_gradient(self, input, backwards_input):
        pass

    @abstractmethod
    def _forward_pass(self, input):
        pass

    @abstractmethod
    def _check_input_dim(self, input):
        pass

    @abstractmethod
    def _check_backwards_input_dim(self, input):
        pass

    def save_current_input(self, input):
        self._current_input = input

    def save_current_backward_input(self, backward_input):
        self._current_backward_input = backward_input

    def forward(self, input):
        self._check_input_dim(input=input)
        self.save_current_input(input)
        return self._forward_pass(input)

    def backwards(self, backwards_input):
        self.save_current_backward_input(backward_input=backwards_input)
        return self._backward_gradient(input=self._current_input, backwards_input=backwards_input)

