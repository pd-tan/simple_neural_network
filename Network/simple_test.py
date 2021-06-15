from Layers.SigmoidLayer import SigmoidLayer
from Layers.ReLU1D import ReLU1D
from Layers.FullyConnectedLayer1D import FullyConnectedLayer1D
from Layers.BatchNorm1D import BatchNorm1D
from WeightInitialisation.WeightInitialisationTypes import WeightInitialisationType
from DataGeneration.DataGenerator import DataGenerator
from Layers.CrossEntropyLoss import CrossEntropyLoss
from Layers.BaseClasses.TrainableLayersABC import TrainableStandardLayersABC
import numpy as np

np.random.seed()

batch_size = 4

L_1 = FullyConnectedLayer1D(input_length=32, output_length=16, batch_size=batch_size,
                            weight_init_method=WeightInitialisationType.HE)
L_1_BN = BatchNorm1D(input_length=16, batch_size=batch_size)
L_1_A = ReLU1D(input_length=16, batch_size=batch_size)

L_2 = FullyConnectedLayer1D(input_length=16, output_length=16, batch_size=batch_size,
                            weight_init_method=WeightInitialisationType.HE)
L_2_BN = BatchNorm1D(input_length=16, batch_size=batch_size)
L_2_A = ReLU1D(input_length=16, batch_size=batch_size)

L_3 = FullyConnectedLayer1D(input_length=16, output_length=2, batch_size=batch_size,
                            weight_init_method=WeightInitialisationType.HE)
L_3_A = SigmoidLayer(input_length=2, batch_size=batch_size)

CELoss = CrossEntropyLoss(input_length=2, batch_size=batch_size)
layers = [L_1, L_1_BN, L_1_A, L_2, L_2_BN, L_2_A, L_3, L_3_A]
loss_layer = CELoss
step_size = 0.001
loss_sum = 0

number_of_runs = 100000000
print_per_runs = 1000
generator = DataGenerator(32)

val_size = 1000

for run in range(number_of_runs):

    data, truth = generator.generate_data(batch_size=batch_size)
    layer_input = data
    for layer in layers:
        layer_input = layer.forward(layer_input)
    loss = CELoss.forward(layer_input, truth)

    backward_input = CELoss.backwards()

    for layer in layers[::-1]:
        backward_input = layer.backwards(backward_input)
    for layer in layers:
        if isinstance(layer, TrainableStandardLayersABC):
            layer.train(step_size * loss)
    loss_sum = loss_sum + loss
    if run % print_per_runs == 0:
        print("progress:", run / number_of_runs)
        print("loss:", loss_sum / print_per_runs)
        loss_sum = 0
        score_sum = 0

        for val_counter in range(val_size):
            data, truth = generator.generate_data(batch_size=batch_size)
            layer_input = data
            for layer in layers:
                layer_input = layer.forward(layer_input)
            number_correct = np.sum(np.argmax(layer_input, axis=1) == np.argmax(truth, axis=1))
            score_sum = score_sum + number_correct
        percentage_correct = score_sum / (val_size * batch_size)
        print("Percentage_correct:", percentage_correct)
