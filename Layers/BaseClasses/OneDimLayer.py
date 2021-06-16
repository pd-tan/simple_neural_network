from Layers.BaseClasses.StandardLayersABC import StandardLayersABC


class OneDimStandardLayers(StandardLayersABC):
    def __init__(self, input_length, batch_size, output_length):
        assert (input_length != 0), "Input must not be 0"
        assert (batch_size != 0), "Batch size must not be 0"
        self._input_length = input_length
        self._batch_size = batch_size
        self._output_length = output_length

    def _check_input_dim(self, forward_input):
        # TODO add catching of batch * input and convert to batch*input*1 and give warning

        # assertion for each dimension executed seperately for individual error messages
        assert (len(forward_input.shape) == 3), "Input must be of shape `batch size * input length * 1`"
        assert (forward_input.shape[
                    0] == self._batch_size), "The first dimension of your input should be the specified batch size"
        assert (forward_input.shape[
                    1] == self._input_length), "The second dimension of your input should be the input length specified"
        assert (forward_input.shape[2] == 1), "This is a 1D layer, the third dimension of your input should be 1"

    def _check_backwards_input_dim(self, backwards_input):
        assert (len(backwards_input.shape) == 3), "Backwards input must be of shape `batch size * input length * 1`"
        assert (backwards_input.shape[
                    0] == self._batch_size), "The first dimension of your backwards input should be the specified batch size"
        assert (backwards_input.shape[
                    1] == self._output_length), "The second dimension of your backwards input should be the output length specified"
        assert (backwards_input.shape[2] == 1), "This is a 1D layer, the third dimension of your backwards input should be 1"
