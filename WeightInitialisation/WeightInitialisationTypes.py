from enum import Enum
from WeightInitialisation.HeInitialisation import HeInit
from WeightInitialisation.ZeroInitialisation import ZeroInit
from WeightInitialisation.RandomInitialisation import RandomInit
class WeightInitialisationType(Enum):
    ZERO = 1
    RANDOM = 2
    HE = 3

