from abc import ABCMeta, abstractmethod


class LossLayersABC(metaclass=ABCMeta):

    @abstractmethod
    def backward_gradient(self, input, truth):
        pass

    @abstractmethod
    def forward_pass(self, input, truth):
        pass

    @abstractmethod
    def check_input_dim(self, input):
        pass

    @abstractmethod
    def check_truth_dim(self, truth):
        pass

    def save_current_input(self, input):
        self._current_input = input

    def save_current_truth(self, current_truth):
        self._current_truth = current_truth

    def forward(self, input,truth):
        self.check_input_dim(input=input)
        self.check_truth_dim(truth=input)
        self.save_current_input(input)
        self.save_current_truth(truth)
        return self.forward_pass(input,truth)

    def backwards(self):
        return self.backward_gradient(input=self._current_input, truth=self._current_truth)

