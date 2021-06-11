import unittest
from Layers.ReLU1D import ReLU1D
import numpy as np

class ReLU_value_test(unittest.TestCase):
    def test_output_value_non_zero(self):
        for batch_size in range(1,10):
            for input_dim in range(1,100):
                test_layer = ReLU1D(input_dim,batch_size)
                input_template = np.ones((batch_size,input_dim,1))
                for value in np.linspace(0.1,100,300):
                    input = input_template*value
                    self.assertTrue((input == test_layer.forward(input)).all())

    def test_output_value_zero(self):
        for batch_size in range(1, 10):
            for input_dim in range(1, 100):
                test_layer = ReLU1D(input_dim, batch_size)
                input_template = np.ones((batch_size, input_dim, 1))
                zero_arr = input_template*0
                for value in np.linspace(-100, 0, 300):
                    input = input_template * value
                    self.assertTrue((zero_arr == test_layer.forward(input)).all())
    def test_gradient_value_input_greater_zero(self):
        for batch_size in range(1, 10):
            for input_dim in range(1, 100):
                test_layer = ReLU1D(input_dim, batch_size)
                input = np.random.randn(batch_size,input_dim,1)
                input = np.maximum(input,0.0001)
                backward_input = np.random.randn(batch_size,input_dim,1)
                test_layer.forward(input)
                self.assertTrue((backward_input == test_layer.backwards(backward_input)).all())
    def test_gradient_value_input_lesser_zero(self):
        for batch_size in range(1, 10):
            for input_dim in range(1, 100):
                test_layer = ReLU1D(input_dim, batch_size)
                input = np.random.randn(batch_size,input_dim,1)
                input = np.minimum(input,0)
                backward_input = np.random.randn(batch_size,input_dim,1)
                test_layer.forward(input)
                self.assertTrue((np.zeros_like(input) == test_layer.backwards(backward_input)).all())



if __name__ == '__main__':
    unittest.main()
