import unittest
from WeightInitialisation.ZeroInitialisation import ZeroInit
class ZeroInitTestCases(unittest.TestCase):
    def test_dim(self):
        for input_length in range(1,100):
            for output_length in range (1,100):
                self.assertEqual((output_length,input_length),ZeroInit(input_length=input_length, output_length=output_length).shape)



if __name__ == '__main__':
    unittest.main()
