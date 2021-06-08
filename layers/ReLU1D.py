from layers.LayersABC import LayerABC

class ReLU1D(LayerABC):
    def __init__(self, input_dim):
        self._input_dim = input_dim
    def forward