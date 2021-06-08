from abc import ABCMeta, abstractmethod

class LayerABC(metaclass=ABCMeta):
    def __init__(self):
        self.__weights = None

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def back(self):
        pass

    @abstractmethod
    def init_weights(self):
        pass

    @abstractmethod
    def has_weights(self):
        pass

