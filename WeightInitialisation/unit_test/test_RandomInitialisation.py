import unittest
from WeightInitialisation.RandomInitialisation import RandomInit
class RandomInitTestCases(unittest.TestCase):
    def test_dim(self):
        for batch_size in range(1, 5):
            for input_length in range(1, 100):
                for output_length in range(1, 100):
                    self.assertEqual((batch_size, output_length, input_length),
                                     RandomInit(input_length=input_length, output_length=output_length,
                                            batch_size=batch_size).shape)


if __name__ == '__main__':
    unittest.main()
