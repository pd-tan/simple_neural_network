from Layers.LayersABC import LayerABC
from Layers.SigmoidLayer import SigmoidLayer
from Layers.ReLU1D import ReLU1D
from Layers.FullyConnectedLayer1D import FullyConnectedLayer1D
from Layers.BatchNorm1D import BatchNorm1D


L_1 = FullyConnectedLayer1D(16,16)
L_1_BN = BatchNorm1D(16)
L_1_A = ReLU1D()
