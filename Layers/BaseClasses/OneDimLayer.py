from Layers.BaseClasses.LayersABC import LayerABC


class OneDimLayer(LayerABC):
    def __init__(self, input_length):
        assert (input_length != 0), "Input must not be 0"
        self._input_length = input_length

    def forward(self, input):
        assert (len(input.shape) == 1), "Input must be one-dimensional"
