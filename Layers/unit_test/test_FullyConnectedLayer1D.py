import unittest
from Layers.FullyConnectedLayer1D import FullyConnectedLayer1D
import numpy as np


class FullyConnectedDimTest(unittest.TestCase):
    test_layer_0 = FullyConnectedLayer1D(input_length=0, output_length=15)
    test_layer_1 = FullyConnectedLayer1D(input_length=1, output_length=14)
    test_layer_2 = FullyConnectedLayer1D(input_length=2, output_length=13)

    def test_forward_input_dim(self):
        with self.assertRaises(ValueError):
            self.test_layer_0.forward(np.zeros((3, 1)))
        with self.assertRaises(ValueError):
            self.test_layer_1.forward(np.zeros((3, 1)))
        with self.assertRaises(ValueError):
            self.test_layer_2.forward(np.zeros((3, 1)))

    def test_forward_output_dim(self):
        self.assertEqual(len(self.test_layer_1.forward(np.zeros((1, 1)))), 14)
        self.assertEqual(len(self.test_layer_2.forward(np.zeros((2, 1)))), 13)


if __name__ == '__main__':
    unittest.main()
