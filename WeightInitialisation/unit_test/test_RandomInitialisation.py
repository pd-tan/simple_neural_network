import unittest
from WeightInitialisation.RandomInitialisation import RandomInit
class RandomInitTestCases(unittest.TestCase):
    def test_dim(self):
        for input_length in range(1,100):
            for output_length in range (1,100):
                self.assertEqual((output_length,input_length),RandomInit(input_length=input_length, output_length=output_length).shape)



if __name__ == '__main__':
    unittest.main()
