import unittest
import numpy as np
from Layers.BatchNorm1D import BatchNorm1D


class BatchNormValueTest(unittest.TestCase):
    def test_something(self):
        for batch_size in range(1,10):
            for input_length in range(1,20):
                input_data = np.random.randn(batch_size,input_length,1)
                test_layer = BatchNorm1D(input_length=input_length, batch_size=batch_size)
                output_data = test_layer.forward(input_data)
                print()
                self.assertTrue(np.abs(np.sum(output_data,axis=0)).all()<0.1)



if __name__ == '__main__':
    unittest.main()
