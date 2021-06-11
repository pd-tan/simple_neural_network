import numpy as np

from Layers.BaseClasses.OneDimLayer import OneDimLayer

class CrossEntropyLoss(OneDimLayer):
    def __init__(self,input_length,batch_size):
        super().__init__(input_length=input_length, batch_size=batch_size)
        pass
    def forward(self,input,truth):
        super().check_dim(input)
        super().check_dim(truth)
        loss = -(truth*np.log(input) + (1-truth)*np.log(1-input))
        loss = np.sum(loss,axis=1)
        return loss

    def back(self,input,truth,loss):
        return -(truth/input+(1-truth)*-1/(1-input))
        pass


if __name__ == '__main__':
    from Layers.SigmoidLayer import SigmoidLayer
    print("Simple test of CELoss Layer");
    test_layer = CrossEntropyLoss(3,10)
    sigmod = SigmoidLayer(3,10)
    network_output = np.random.randn(10,3,1)
    truth = np.round(np.random.uniform(0,1,(10,3,1)))
    print(sigmod.forward(network_output))
    print(truth)
    loss = test_layer.forward(sigmod.forward(network_output),truth)
    print(loss)
    print(test_layer.back(network_output,truth,loss))



