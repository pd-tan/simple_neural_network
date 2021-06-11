from abc import ABCMeta, abstractmethod

class LayerABC(metaclass=ABCMeta):
    def __init__(self,input_length,batch_size):
        pass

    @abstractmethod
    def forward_pass(self,input):
        pass

    @abstractmethod
    def gradient(self, backwards_input):
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
        self.forward_pass(input)

    def back(self, backwards_input):
        self.save_current_backward_input(backward_input=backwards_input)
        return self.gradient(backwards_input=backwards_input)


