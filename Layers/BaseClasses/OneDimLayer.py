from Layers.BaseClasses.LayersABC import LayerABC


class OneDimLayer(LayerABC):
    def __init__(self, input_length):

        self._input_length = input_length

    def forward(self, input):
        assert (len(input.shape) == 2)
        assert (input.shape[1] == 1)
