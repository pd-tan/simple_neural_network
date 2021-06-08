from layers.LayersABC import LayerABC


class ReLU1D(LayerABC):
    def __init__(self, input_dim):
        self._input_dim = input_dim

    def forward(self, input):
        return input * (input > 1)


if __name__ == '__main__':
    print("Simple test of ReLU1D Layer");
    test_layer = ReLU1D()
    print(test_layer.forward(np.random.rand(16, 1)))
