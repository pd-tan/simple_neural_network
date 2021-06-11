from abc import ABCMeta, abstractmethod

class LayerABC(metaclass=ABCMeta):
    def __init__(self,input_length,batch_size):
        pass

    @abstractmethod
    def forward(self,input):
        pass

    @abstractmethod
    def back(self,input,backwards_input):
        pass

