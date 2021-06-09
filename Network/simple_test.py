from Layers.SigmoidLayer import SigmoidLayer
from Layers.ReLU1D import ReLU1D
from Layers.FullyConnectedLayer1D import FullyConnectedLayer1D
from Layers.BatchNorm1D import BatchNorm1D
from WeightInitialisation.WeightInitialisationTypes import WeightInitialisationType
import numpy as np
np.random.seed()
L_1 = FullyConnectedLayer1D(32,16,4,weight_init_method=WeightInitialisationType.HE)
L_1_BN = BatchNorm1D(16,4)
L_1_A = ReLU1D(16,4)

L_2 = FullyConnectedLayer1D(16,16,4,weight_init_method=WeightInitialisationType.HE)
L_2_BN = BatchNorm1D(16,4)
L_2_A = ReLU1D(16,4)

L_3 = FullyConnectedLayer1D(16,2,4,weight_init_method=WeightInitialisationType.HE)
L_3_A = SigmoidLayer(2,4)

input = np.random.randn(4,32,1)
output = np.argmax(L_3_A.forward(L_3.forward(L_2_BN.forward(L_2_BN.forward(L_2.forward(L_1_BN.forward(L_1_BN.forward(L_1.forward(input)))))))),axis=1)
print("input is: ", input)
print("output is: ", output)
