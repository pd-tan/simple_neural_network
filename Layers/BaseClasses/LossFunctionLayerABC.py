from abc import ABCMeta, abstractmethod


class LossLayersABC(metaclass=ABCMeta):

    @abstractmethod
    def backward_gradient(self, forward_input, truth):
        pass

    @abstractmethod
    def forward_pass(self, forward_input, truth):
        pass

    @abstractmethod
    def check_input_dim(self, forward_input):
        pass

    @abstractmethod
    def check_truth_dim(self, truth):
        pass

    def save_current_input(self, forward_input):
        self._current_forward_input = forward_input

    def save_current_truth(self, current_truth):
        self._current_truth = current_truth

    def forward(self, input,truth):
        self.check_input_dim(forward_input=input)
        self.check_truth_dim(truth=input)
        self.save_current_input(input)
        self.save_current_truth(truth)
        return self.forward_pass(input,truth)

    def backwards(self):
        return self.backward_gradient(forward_input=self._current_forward_input, truth=self._current_truth)

