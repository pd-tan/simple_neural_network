import unittest
from Layers.FullyConnectedLayer1D import FullyConnectedLayer1D
import numpy as np


class FullyConnectedForwardTest(unittest.TestCase):

    test_layer_1 = FullyConnectedLayer1D(input_length=1, output_length=14)
    test_layer_2 = FullyConnectedLayer1D(input_length=2, output_length=13)

    def test_forward_input_dim(self):
        with self.assertRaises(AssertionError):
            self.test_layer_1.forward(np.zeros((1, 1)))
        with self.assertRaises(AssertionError):
            self.test_layer_2.forward(np.zeros((2, 1)))

    def test_forward_input_length(self):
        for desired_input_length in range(1,100):
            for actual_input_length in range(1,100):
                if actual_input_length == desired_input_length:
                    pass
                else:
                    test_layer = FullyConnectedLayer1D(input_length=desired_input_length, output_length=1)
                    with self.assertRaises(ValueError):
                        test_layer.forward(np.zeros(actual_input_length))

    def test_forward_output_dim(self):
        for input_length in range(1,100):
            for output_length in range(1,100):
                test_layer = FullyConnectedLayer1D(input_length=input_length,output_length=output_length)
                self.assertEqual((output_length,), test_layer.forward(np.zeros(input_length)).shape)



if __name__ == '__main__':
    unittest.main()
