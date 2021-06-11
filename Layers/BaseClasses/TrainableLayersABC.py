from Layers.BaseClasses.StandardLayersABC import StandardLayersABC
from abc import abstractmethod
class TrainableStandardLayersABC(StandardLayersABC):

    @abstractmethod
    def init_weights(self,weight_init_method,bias_init_method):
        pass

    @abstractmethod
    def update_weight(self):
        pass

