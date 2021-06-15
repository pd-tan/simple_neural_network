import numpy as np


class DataGenerator():
    def __init__(self, input_size):
        self._input_size = input_size
        self._number_of_labels = 2
        self._threshold = 0
        self._weightage_vector = np.random.randint(low=-4, high=4, size=(self._input_size, 1))

    def generate_data(self, batch_size):

        input_data = np.random.randn(batch_size, self._input_size, 1)
        weighted_values = input_data * self._weightage_vector
        summed_values = np.sum(weighted_values, axis=1)
        label_switch = summed_values > self._threshold
        truth = np.zeros((batch_size, self._number_of_labels, 1))
        # TODO find faster approach to achieve this
        for sample_number in range(batch_size):
            if label_switch[sample_number, 0]:
                truth[sample_number, 0, 0] = 1
            else:
                truth[sample_number, 1, 0] = 1

        return input_data, truth


if __name__ == '__main__':
    generator = DataGenerator(16)
    generator.generate_data(4)
    generator.generate_data(4)
    generator.generate_data(4)
    generator.generate_data(4)
    generator.generate_data(4)
    generator.generate_data(4)
    generator.generate_data(4)
    generator.generate_data(4)
