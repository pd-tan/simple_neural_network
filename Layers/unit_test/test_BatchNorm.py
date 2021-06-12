import unittest
import numpy as np
from Layers.BatchNorm1D import BatchNorm1D


class BatchNormValueTest(unittest.TestCase):
    def test_output_value(self):
        for batch_size in range(2, 10):
            for input_length in range(1, 20):
                beta = np.random.randn(input_length, 1)
                eps = np.random.rand(input_length, 1)
                input_data = np.random.randn(batch_size, input_length, 1)
                test_layer = BatchNorm1D(input_length=input_length, batch_size=batch_size, beta=beta, eps=eps)
                output_data = test_layer.forward(input_data)
                output_mean = np.mean(output_data, axis=0)
                self.assertTrue(np.allclose(np.var(output_data, axis=0), np.var(input_data, axis=0) / (
                        np.var(input_data, axis=0) + test_layer._eps)))

                # TODO ALter params of all close to desired tolerance
                self.assertTrue(np.allclose(output_mean, test_layer._beta))


if __name__ == '__main__':
    unittest.main()
