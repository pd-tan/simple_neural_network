from enum import Enum
from WeightInitialisation.HeInitialisation import HeInit
from WeightInitialisation.ZeroInitialisation import ZeroInit
from WeightInitialisation.RandomInitialisation import RandomInit


class WeightInitialisationType(Enum):
    ZERO = 1
    RANDOM = 2
    HE = 3


def WeightInitialiser(init_type, input_length, output_length):
    switcher = {
        WeightInitialisationType.ZERO: ZeroInit,
        WeightInitialisationType.RANDOM: RandomInit,
        WeightInitialisationType.HE: HeInit

    }
    init_func = switcher.get(init_type)
    return init_func(input_length=input_length,output_length=output_length)
