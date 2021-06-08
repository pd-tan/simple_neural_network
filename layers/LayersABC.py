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

