from Layers.SigmoidLayer import SigmoidLayer
from Layers.ReLU1D import ReLU1D
from Layers.FullyConnectedLayer1D import FullyConnectedLayer1D
from Layers.BatchNorm1D import BatchNorm1D
from WeightInitialisation.WeightInitialisationTypes import WeightInitialisationType
from DataGeneration.DataGenerator import DataGenerator
from Layers.CrossEntropyLoss import CrossEntropyLoss
from Layers.BaseClasses.TrainableLayersABC import TrainableStandardLayersABC
import numpy as np
import matplotlib.pyplot as plt

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
step_size = 0.00005
training_loss_sum = 0
validation_loss_sum = 0
score_sum = 0
number_of_runs = 50000
epoch_size = 1000
generator = DataGenerator(32)

val_size = 200
training_loss_list = []
validation_loss_list = []
accuracy_list = []

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
    training_loss_sum = training_loss_sum + loss
    if run % epoch_size == 0 and run != 0:
        training_loss = training_loss_sum / epoch_size

        print("progress:", run / number_of_runs)

        print("training loss:", training_loss)

        # Run validation
        for val_counter in range(val_size):
            data, truth = generator.generate_data(batch_size=batch_size)
            layer_input = data
            for layer in layers:
                layer_input = layer.forward(layer_input)

            validation_loss_sum = validation_loss_sum + CELoss.forward(layer_input, truth)
            number_correct = np.sum(np.argmax(layer_input, axis=1) == np.argmax(truth, axis=1))
            score_sum = score_sum + number_correct
        accuracy = score_sum / (val_size * batch_size)
        validation_loss = validation_loss_sum / val_size
        print("validation Loss: ", validation_loss)
        print("Accuracy:", accuracy)
        print("__________________________________")
        training_loss_list.append(training_loss)
        validation_loss_list.append(validation_loss)
        accuracy_list.append(accuracy)
        training_loss_sum = 0
        validation_loss_sum = 0
        score_sum = 0
fig, axs = plt.subplots(3)
plt.subplot(3,1,1)
plt.ylabel('Training Loss')
plt.subplot(3,1,2)
plt.ylabel('Validation Loss')
plt.subplot(3,1,3)
plt.ylabel('Validation Accuracy')
axs[0].plot(list(range(0, len(training_loss_list))), training_loss_list)
axs[1].plot(list(range(0, len(validation_loss_list))), validation_loss_list)
axs[2].plot(list(range(0, len(accuracy_list))), accuracy_list)
plt.show()
