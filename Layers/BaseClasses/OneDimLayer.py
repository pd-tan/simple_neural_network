from Layers.BaseClasses.LayersABC import LayerABC


class OneDimLayer(LayerABC):
    def __init__(self, input_length, batch_size):
        assert (input_length != 0), "Input must not be 0"
        assert (batch_size != 0), "Batch size must not be 0"
        self._input_length = input_length
        self._batch_size = batch_size

    def save_current_input(self, input):
        self._current_input = input

    def save_current_backward_input(self, backward_input):
        self._current_backward_input = backward_input

    def check_dim(self, input):
        # TODO add catching of batch * input and convert to batch*input*1 and give warning
        assert (len(input.shape) == 3), "Input must be one-dimensional x batch size"
        assert (input.shape[0] == self._batch_size), "The first dimension of your input should be the speciified batch size"
        assert (input.shape[1] == self._input_length),"The second dimension of your input should be the input length specified"
        assert (input.shape[2] == 1), "This is a 1D layer, the third dimension of your input should be 1"
