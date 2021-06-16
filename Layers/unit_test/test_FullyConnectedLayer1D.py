import unittest
from Layers.FullyConnectedLayer1D import FullyConnectedLayer1D
import numpy as np


class FullyConnectedForwardTest(unittest.TestCase):

    def test_forward_input_length(self):
        for output_length in range(1, 10):
            for desired_batch_size in range(1, 10):
                for actual_batch_size in range(1, 10):
                    for desired_input_length in range(1, 10):
                        for actual_input_length in range(1, 10):
                            if actual_input_length == desired_input_length and actual_batch_size == desired_batch_size:
                                pass
                            else:
                                test_layer = FullyConnectedLayer1D(input_length=desired_input_length, output_length=1,
                                                                   batch_size=desired_batch_size)
                                with self.assertRaises(AssertionError):
                                    test_layer.forward(np.zeros((actual_batch_size,actual_input_length,1)))

    def test_forward_output_dim(self):
        for batch_size in range(1, 10):
            for input_length in range(1, 50):
                for output_length in range(1, 50):
                    test_layer = FullyConnectedLayer1D(input_length=input_length, output_length=output_length,batch_size=batch_size)
                    self.assertEqual((batch_size,output_length,1), test_layer.forward(np.zeros((batch_size,input_length,1))).shape)



if __name__ == '__main__':
    unittest.main()
