from Layers.BaseClasses.StandardLayersABC import StandardLayersABC
from abc import abstractmethod
class TrainableStandardLayersABC(StandardLayersABC):

    @abstractmethod
    def init_weights(self,weight_init_method,bias_init_method):
        pass

    @abstractmethod
    def update_weights(self, step_size):
        pass

    @abstractmethod
    def calculate_weights_gradient(self,input,backwards_input):
        pass

    @abstractmethod
    def check_weights_gradient_dim(self):
        pass

    def train(self,step_size):
        self.calculate_weights_gradient(self._current_input,self._current_backward_input)
        self.check_weights_gradient_dim()
        self.update_weights(step_size)

