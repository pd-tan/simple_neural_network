import unittest
from WeightInitialisation.HeInitialisation import HeInit

class HeInitTestCases(unittest.TestCase):
    def test_dim(self):
        for batch_size in range(1, 5):
            for input_length in range(1,100):
                for output_length in range (1,100):
                    self.assertEqual((batch_size,output_length,input_length),HeInit(input_length=input_length, output_length=output_length, batch_size=batch_size).shape)



if __name__ == '__main__':
    unittest.main()
