import numpy as np

from Layers.BaseClasses.OneDimLayer import OneDimLayer

class CrossEntropyLoss(OneDimLayer):
    def __init__(self,input_length,batch_size):
        super().__init__(input_length=input_length, batch_size=batch_size)
        pass
    def forward(self,input,truth):
        super().forward(input)
        


