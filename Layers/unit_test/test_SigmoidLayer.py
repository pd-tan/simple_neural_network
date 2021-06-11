import unittest
from Layers.SigmoidLayer import SigmoidLayer
import numpy as np

class SigmoidValueTest(unittest.TestCase):
    def test_sigmoid_output_value(self):
        for batch_size in range(1,10):
            for input_dim in range(1,100):
                test_layer = SigmoidLayer(input_dim,batch_size)
                input_template = np.ones((batch_size,input_dim,1))
                for value in np.linspace(-100,100,300):
                    input = input_template*value
                    output = input_template*1 / (1 + np.exp(-value))
                    self.assertTrue((output == test_layer.forward(input)).all())


if __name__ == '__main__':
    unittest.main()
