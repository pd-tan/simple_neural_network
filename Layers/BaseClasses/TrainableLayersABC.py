from Layers.BaseClasses.LayersABC import LayerABC
from abc import abstractmethod
class TrainableLayersABC(LayerABC):
    def __init__(self):
        pass
    @abstractmethod
    def init_weights(self,weight_init_method,bias_init_method):
        pass

    @abstractmethod
    def update_weight(self):
        pass

