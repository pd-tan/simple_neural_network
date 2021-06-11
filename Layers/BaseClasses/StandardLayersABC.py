from abc import ABCMeta, abstractmethod


class StandardLayersABC(metaclass=ABCMeta):

    @abstractmethod
    def backward_gradient(self, input, backwards_input):
        pass

    @abstractmethod
    def forward_pass(self, input):
        pass

    @abstractmethod
    def check_input_dim(self, input):
        pass

    @abstractmethod
    def check_backwards_input_dim(self, input):
        pass

    def save_current_input(self, input):
        self._current_input = input

    def save_current_backward_input(self, backward_input):
        self._current_backward_input = backward_input

    def forward(self, input):
        self.check_input_dim(input=input)
        self.save_current_input(input)
        return self.forward_pass(input)

    def backwards(self, backwards_input):
        self.save_current_backward_input(backward_input=backwards_input)
        return self.backward_gradient(input=self._current_input, backwards_input=backwards_input)

