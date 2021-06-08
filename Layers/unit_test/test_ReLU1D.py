import unittest
from Layers.ReLU1D import ReLU1D
import numpy as np

class ReLU_value_test(unittest.TestCase):
    def test_non_zero(self):
        for input_dim in range(1,100):
            test_layer = ReLU1D(input_dim)
            input_template = np.ones(input_dim)
            for value in np.linspace(0.1,100,300):
                input = input_template*value
                self.assertTrue((input == test_layer.forward(input)).all())

    def test_zero(self):
        for input_dim in range(1, 100):
            test_layer = ReLU1D(input_dim)
            input_template = np.ones(input_dim)
            zero_arr = np.zeros(input_dim)
            for value in np.linspace(-100,0, 300):
                input = input_template * value
                self.assertTrue((zero_arr== test_layer.forward(input)).all())



if __name__ == '__main__':
    unittest.main()
